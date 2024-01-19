import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ElectraModel, ElectraConfig, DataCollatorForLanguageModeling, BertForMaskedLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import argparse


def get_dataloaders(tokenizer):
  dataset = load_dataset("papluca/language-identification")

  def preprocess_function(examples):
      return tokenizer(examples['text'], add_special_tokens=True)

  print(len(dataset['train']))

  n_samples = 100  # the number of training example

  # We first shuffle the data !
  dataset = dataset.shuffle()

  # Select n_samples
  train_dataset = dataset['train'].select(range(n_samples))
  valid_dataset = dataset['validation'].select(range(n_samples//5))

  # Tokenize the dataset
  train_dataset = train_dataset.map(
      preprocess_function, remove_columns=dataset["train"].column_names, batched=True, 
  )
  valid_dataset = valid_dataset.map(
      preprocess_function, remove_columns=dataset["train"].column_names, batched=True,
  )

  block_size = 128
  def group_texts(examples): # From HF MLM Sample
      # Concatenate all texts.
      concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
      total_length = len(concatenated_examples[list(examples.keys())[0]])
      # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
      # customize this part to your needs.
      if total_length >= block_size:
          total_length = (total_length // block_size) * block_size
      # Split by chunks of block_size.
      result = {
          k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
          for k, t in concatenated_examples.items()
      }
      return result
  train_dataset = train_dataset.map(group_texts, batched=True)
  valid_dataset = valid_dataset.map(group_texts, batched=True)
  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=.15)

  batch_size = 32

  train_dataloader = DataLoader(
      train_dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True, pin_memory_device='cuda:0'
  )
  valid_dataloader = DataLoader(
      valid_dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True, pin_memory_device='cuda:0'
  )
  return train_dataloader, valid_dataloader
  

class DiscriminatorHead(nn.Module):
    """Discriminator module for the generator, made up of two dense layers."""
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.dense2 = nn.Linear(config.embedding_size, 2)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = torch.nn.GELU()(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense2(hidden_states)

        return hidden_states

class Discriminator(nn.Module):
    """Complete Discriminator"""
    def __init__(self, discriminator_body, discriminator_head):
        super().__init__()
        self.discriminator_body = discriminator_body
        self.discriminator_head = discriminator_head

    def forward(self, input):
      output = self.discriminator_body(input).last_hidden_state
      output = self.discriminator_head(output)
      return output
    

def train(generator, discriminator, n_epochs, train_dataloader, valid_dataloader, tokenizer, run_name, lr=5e-5):
    optimizer_gen = torch.optim.AdamW(
        generator.parameters(),
        lr=lr,
        eps=1e-08,
    )
    optimizer_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=lr,
        eps=1e-08,
    )
    gen_loss = nn.CrossEntropyLoss(ignore_index = -100)
    discrim_loss = nn.BCEWithLogitsLoss()
    list_train_loss = []
    list_val_loss = []
    list_val_gen_acc = []
    list_val_dis_acc = []
    best_val_loss = 10**50
    generator.cuda()
    discriminator.cuda()
    for e in range(n_epochs):
        # ========== Training ==========

        # Set model to training mode
        generator.train()
        discriminator.train()

        # Tracking variables
        train_loss = 0
        for batch in tqdm(train_dataloader):
            input_ids, attention_mask, labels =(
                batch["input_ids"].cuda(),
                batch["attention_mask"].cuda(),
                batch["labels"].cuda(),
            )
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()

            generator_pred = generator(input_ids, attention_mask)
            loss_g = gen_loss(generator_pred.logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            
            
            predicted_tokens = torch.argmax(generator_pred.logits, dim=-1)
            discriminator_input = predicted_tokens
            discriminator_predictions = discriminator(discriminator_input)
            discriminator_labels = torch.nn.functional.one_hot((input_ids == predicted_tokens)*1, num_classes=2).float()
            loss_d = discrim_loss(discriminator_predictions, discriminator_labels)

            loss_g.backward()
            loss_d.backward()

            optimizer_gen.step()
            optimizer_disc.step()

            train_loss += loss_g.item() + loss_d.item()
        list_train_loss.append(train_loss / len(train_dataloader))
        # ========== Validation ==========
        if e%1 == 0:
            generator.eval()
            discriminator.eval()
            valid_loss = 0
            val_gen_acc = 0
            val_disc_acc = 0
            for batch in tqdm(valid_dataloader):
                input_ids, attention_mask, labels =(
                    batch["input_ids"].cuda(),
                    batch["attention_mask"].cuda(),
                    batch["labels"].cuda(),
                )
                # Forward pass
            
                generator_pred = generator(input_ids, attention_mask)
                loss_g = gen_loss(generator_pred.logits.view(-1, tokenizer.vocab_size), labels.view(-1))
            
                predicted_tokens = torch.argmax(generator_pred.logits, dim=-1)
                discriminator_input = predicted_tokens
                discriminator_predictions = discriminator(discriminator_input)
                discriminator_labels = (input_ids == predicted_tokens)*1
                loss_d = discrim_loss(discriminator_predictions, torch.nn.functional.one_hot(discriminator_labels, num_classes=2).float())

                valid_loss += loss_g.item() + loss_d.item()
                
                generator_accuracy = torch.mean((predicted_tokens[labels != -100] == labels[labels != -100])*1.)
                discriminator_accuracy = torch.mean((torch.argmax(discriminator_predictions, dim=-1) == discriminator_labels)*1.)
                val_gen_acc += generator_accuracy
                val_disc_acc += discriminator_accuracy
            
            val_gen_acc /= len(valid_dataloader)
            list_val_gen_acc.append(val_gen_acc)
            val_disc_acc /= len(valid_dataloader)
            list_val_dis_acc.append(val_disc_acc)
            list_val_loss.append(valid_loss / len(valid_dataloader))
        if list_val_loss[-1] < best_val_loss:
          best_val_loss = list_val_loss[-1]
          generator_path = f"checkpoints/{run_name}/generator_epoch{e}_lr{lr}"
          discriminator_path =  f"checkpoints/{run_name}/discriminator_epoch{e}_lr{lr}"
          torch.save(generator.state_dict(), generator_path)
          torch.save(discriminator.state_dict(), discriminator_path)

        print(
            e,
            "\n\t - Train loss: {:.4f}".format(list_train_loss[-1]),
            "\n\t - Val loss: {:.4f}".format(list_val_loss[-1]),
            "\n\t - Val gen acc: {:.4f}".format(val_gen_acc),
            "\n\t - Val disc acc: {:.4f}".format(val_disc_acc),
        )
    return list_train_loss, list_val_loss, [float(tensor) for tensor in list_val_gen_acc], [float(tensor) for tensor in list_val_dis_acc]

def run(run_name):
  def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

  folder_path = f'checkpoints/{run_name}'
  create_folder(folder_path)
  folder_path = f'outputs/{run_name}'
  create_folder(folder_path)
  
  model_ckpt = "papluca/xlm-roberta-base-language-detection"
  tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

  generator_config = ElectraConfig(vocab_size = tokenizer.vocab_size, embedding_size= 32, hidden_size = 64)
  generator = BertForMaskedLM(config=generator_config)

  discriminator_config = ElectraConfig(vocab_size = tokenizer.vocab_size, embedding_size=64, hidden_size=128)
  discriminator_body = ElectraModel(discriminator_config)
  discriminator_head = DiscriminatorHead(discriminator_config)
  discriminator = Discriminator(discriminator_body,discriminator_head)

  train_dl, valid_dl = get_dataloaders(tokenizer=tokenizer)

  train_losses, val_losses, gen_acc, disc_acc = train(
      generator=generator, 
      discriminator=discriminator, 
      n_epochs=1, 
      train_dataloader=train_dl, 
      valid_dataloader=valid_dl, 
      tokenizer=tokenizer, 
      run_name=run_name
    )
  
  train_losses_file = f'{folder_path}/train_losses.json'
  val_losses_file = f'{folder_path}/val_losses.json'
  gen_acc_file = f'{folder_path}/gen_acc.json'
  disc_acc_file = f'{folder_path}/disc_acc.json'

  # Save to JSON files
  with open(train_losses_file, 'w') as f:
      json.dump(train_losses, f)

  with open(val_losses_file, 'w') as f:
      json.dump(val_losses, f)

  with open(gen_acc_file, 'w') as f:
      json.dump(gen_acc.to_list(), f)

  with open(disc_acc_file, 'w') as f:
      json.dump(disc_acc.to_list(), f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, help='the run name')
    args = parser.parse_args()
    run_name = args.run_name

    run(run_name)
