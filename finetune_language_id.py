import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ElectraModel, ElectraConfig, DataCollatorForLanguageModeling, BertForMaskedLM, AutoTokenizer, DefaultDataCollator
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import argparse


def get_dataloaders(tokenizer):
  dataset = load_dataset("papluca/language-identification")

  def preprocess_function(examples):
      return tokenizer(examples['text'], add_special_tokens=True)

  # n_samples = 100  # the number of training example

  # We first shuffle the data !
  dataset = dataset.shuffle()

  # Select n_samples
  train_dataset = dataset['validation']
  valid_dataset = dataset['test']

  # Tokenize the dataset
  train_dataset = train_dataset.map(
      preprocess_function, remove_columns=dataset["train"].column_names, batched=True, 
  )
  valid_dataset = valid_dataset.map(
      preprocess_function, remove_columns=dataset["train"].column_names, batched=True,
  )

  data_collator = DefaultDataCollator(tokenizer)

  batch_size = 32

  train_dataloader = DataLoader(
      train_dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True, pin_memory_device='cuda:0'
  )
  valid_dataloader = DataLoader(
      valid_dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True, pin_memory_device='cuda:0'
  )
  return train_dataloader, valid_dataloader
  

class LanguageIdHead(nn.Module):
    """Discriminator module for the generator, made up of two dense layers."""
    def __init__(self, config, num_classes):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.dense2 = nn.Linear(config.embedding_size, num_classes)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = torch.nn.GELU()(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense2(hidden_states)

        return hidden_states
    
class ElectraForLanguageId(nn.Module):
    """Complete Discriminator"""
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head

    def forward(self, input):
      output = self.body(input).last_hidden_state
      output = self.head(output)
      return output

def train(model, n_epochs, train_dataloader, valid_dataloader, run_name, lr=5e-5):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-08,
    )
    loss = nn.CrossEntropyLoss()
    list_train_loss = []
    list_val_loss = []
    list_val_acc = []
    best_val_loss = 10**50
    model.cuda()
    for e in range(n_epochs):
        # ========== Training ==========

        # Set model to training mode
        model.train()

        # Tracking variables
        train_loss = 0
        for batch in tqdm(train_dataloader):
            input_ids, _, labels =(
                batch["input_ids"].cuda(),
                batch["attention_mask"].cuda(),
                batch["labels"].cuda(),
            )
            optimizer.zero_grad()
            
            predictions = model(input_ids)
            loss = loss(predictions, labels)

            loss.backward()
            
            optimizer.step()

            train_loss += loss
        list_train_loss.append(train_loss / len(train_dataloader))
        # ========== Validation ==========
        if True:
            model.eval()
            valid_loss = 0
            val_acc = 0
            for batch in tqdm(valid_dataloader):
                input_ids, _, labels =(
                    batch["input_ids"].cuda(),
                    batch["attention_mask"].cuda(),
                    batch["labels"].cuda(),
                )
                
                predictions = model(input_ids)
                loss = loss(predictions, labels)

                model_accuracy = torch.mean((torch.argmax(predictions, dim=-1) == labels)*1.)
                val_acc += model_accuracy
            
            val_acc /= len(valid_dataloader)
            list_val_loss.append(valid_loss / len(valid_dataloader))
            list_val_acc.append(val_acc)
        
        if list_val_loss[-1] < best_val_loss:
          best_val_loss = list_val_loss[-1]
          model_path =  f"checkpoints/{run_name}/model_epoch{e}_lr{lr}"
          torch.save(model.state_dict(), model_path)

        print(
            e,
            "\n\t - Train loss: {:.4f}".format(list_train_loss[-1]),
            "\n\t - Val loss: {:.4f}".format(list_val_loss[-1]),
            "\n\t - Val acc: {:.4f}".format(val_acc),
        )
    return list_train_loss, list_val_loss, [float(tensor) for tensor in list_val_acc],

def run(run_name, ckpt):
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

    config = ElectraConfig(vocab_size = tokenizer.vocab_size, embedding_size=64, hidden_size=128)
    body = ElectraModel(config)
    body.load_state_dict(torch.load(f'/checkpoints/full_run_1/{ckpt}'))
    head = LanguageIdHead(num_classes=20)
    model = ElectraForLanguageId(body, head)

    train_dl, valid_dl = get_dataloaders(tokenizer=tokenizer)

    train_losses, val_losses, acc = train(
        model=model,
        n_epochs=3, 
        train_dataloader=train_dl, 
        valid_dataloader=valid_dl, 
        run_name=run_name
      )
    
    train_losses_file = f'{folder_path}/train_losses.json'
    val_losses_file = f'{folder_path}/val_losses.json'
    acc_file = f'{folder_path}/gen_acc.json'

    # Save to JSON files
    with open(train_losses_file, 'w') as f:
        json.dump(train_losses, f)

    with open(val_losses_file, 'w') as f:
        json.dump(val_losses, f)

    with open(acc_file, 'w') as f:
        json.dump(acc, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, help='the run name')
    parser.add_argument('checkpoint', type=str, help='checkpoint name')
    
    args = parser.parse_args()
    run_name = args.run_name
    ckpt = args.checkpoint

    run(run_name, ckpt)
