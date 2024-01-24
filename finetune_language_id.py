import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ElectraModel, ElectraConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import argparse

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        
        features = {
            "input_ids": torch.tensor([f["input_ids"] for f in batch]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in batch]),
            "labels": torch.tensor([f["labels"] for f in batch]),
            "masked_ids": torch.tensor([f["masked_ids"] for f in batch]),
        }
        return features


def get_dataloaders(tokenizer):
  dataset = load_dataset("papluca/language-identification")

  # We first shuffle the data !
  dataset = dataset.shuffle()

  # Select n_samples
  train_dataset = dataset['validation']
  valid_dataset = dataset['test']

  labels = {label for label in dataset['validation']['labels']}
  id2label = {idx:label for idx, label in enumerate(labels)}
  label2id = {label:idx for idx, label in enumerate(labels)}

  def preprocess_function(examples):
      encoding = tokenizer(examples['text'], truncation=True, max_length=512)
      encoding['label_ids'] = torch.tensor([label2id[key] for key in examples['labels']])
      encoding['label_ids'] = torch.nn.functional.one_hot(encoding['label_ids'], num_classes=20)
      return encoding

  # Tokenize the dataset
  train_dataset = train_dataset.map(
      preprocess_function, remove_columns=dataset["train"].column_names, batched=True, 
  )
  valid_dataset = valid_dataset.map(
      preprocess_function, remove_columns=dataset["train"].column_names, batched=True,
  )

  data_collator = DataCollatorWithPadding(tokenizer, max_length=512, padding='max_length')

  batch_size = 32

  train_dataloader = DataLoader(
      train_dataset, batch_size=batch_size, collate_fn=data_collator,
  )
  valid_dataloader = DataLoader(
      valid_dataset, batch_size=batch_size, collate_fn=data_collator,
  )
  return train_dataloader, valid_dataloader
  

class LanguageIdHead(nn.Module):
    """Discriminator module for the generator, made up of two dense layers."""
    def __init__(self, config, num_classes):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, num_classes)

    def forward(self, discriminator_hidden_states):
        hidden_states = torch.nn.functional.dropout(hidden_states, .01)
        hidden_states = self.dense(discriminator_hidden_states)

        return hidden_states
    
class ElectraForLanguageId(nn.Module):
    """Complete Discriminator"""
    def __init__(self, body, head):
        super().__init__()
        self.discriminator_body = body
        self.head = head

    def forward(self, input, attention_masks):
      output = self.discriminator_body(input, attention_masks).last_hidden_state
      output = self.head(output)
      return output

def train(model, n_epochs, train_dataloader, valid_dataloader, run_name, lr=5e-5):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-08,
    )
    list_train_loss = []
    list_val_loss = []
    list_val_acc = []
    model.cuda()
    for e in range(n_epochs):
        # ========== Training ==========

        # Set model to training mode
        model.train()
        # Tracking variables
        train_loss = 0
        for batch in tqdm(train_dataloader):
            input_ids, attention_masks, labels =(
                batch["input_ids"].cuda(),
                batch["attention_mask"].cuda(),
                batch["labels"].cuda(),
            )
            optimizer.zero_grad()

            predictions = model(input_ids, attention_masks)
            loss = torch.nn.functional.cross_entropy(predictions, labels.float())

            loss.backward()
            
            optimizer.step()

            train_loss += loss
        list_train_loss.append(float(train_loss / len(train_dataloader)))
        # ========== Validation ==========
        model.eval()
        valid_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                input_ids, attention_masks, labels =(
                    batch["input_ids"].cuda(),
                    batch["attention_mask"].cuda(),
                    batch["labels"].cuda(),
                )
                
                predictions = model(input_ids, attention_masks)
                loss = torch.nn.functional.cross_entropy(predictions, labels.float())
                
                valid_loss += loss
                
                model_accuracy = torch.mean((torch.argmax(predictions, dim=0) == labels)*1.)
                val_acc += model_accuracy
            
                val_acc /= len(valid_dataloader)
                list_val_loss.append(float(valid_loss / len(valid_dataloader)))
                list_val_acc.append(float(val_acc))

        print(
            e,
            "\n\t - Train loss: {:.4f}".format(list_train_loss[-1]),
            "\n\t - Val loss: {:.4f}".format(list_val_loss[-1]),
            "\n\t - Val acc: {:.4f}".format(val_acc),
        )
    return [float(tensor) for tensor in list_train_loss], [float(tensor) for tensor in list_val_loss], [float(tensor) for tensor in list_val_acc],

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

    config = ElectraConfig(vocab_size = tokenizer.vocab_size, embedding_size=64, hidden_size=128)
    body = ElectraModel(config)
    head = LanguageIdHead(config, num_classes=20)
    model = ElectraForLanguageId(body, head)
    model.load_state_dict(torch.load(f'./checkpoints/full_run_2/discriminator_epoch535_lr5e-05'), strict=False)

    train_dl, valid_dl = get_dataloaders(tokenizer=tokenizer)

    train_losses, val_losses, acc = train(
        model=model,
        n_epochs=10, 
        train_dataloader=train_dl, 
        valid_dataloader=valid_dl, 
        run_name=run_name
      )
    
    model_path = f"checkpoints/{run_name}/model_epoch_10"
    torch.save(model.state_dict(), model_path)

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
    
    args = parser.parse_args()
    run_name = args.run_name

    run(run_name)
