#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:56:36 2022

@author: guillermogarcia
"""

import json
import pandas as pd
import numpy as np
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.optim as optim

from preprocess import create_labels

MAX_LEN = 200
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-06


########### DATA FUNCTIONS #############

def load_data():
  with open('train_data_2021.json', 'r') as f:
    data_train_json = json.load(f)
  data_train_pandas = pd.json_normalize(data_train_json)

  with open('taxonomy_mappings_2021.json', 'r') as f:
    taxonomy_mappings = json.load(f)

  return data_train_pandas, taxonomy_mappings

############ BERT FUNCTIONS ##############

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def get_data_loaders(data_train_pandas, tokenizer, percen = 0.8,debug = False):

  train_size = percen
  train_dataset = data_train_pandas.sample(frac=train_size,random_state=200)
  val_dataset = data_train_pandas.drop(train_dataset.index).reset_index(drop=True)
  train_dataset = train_dataset.reset_index(drop=True)

  if debug:
    print("FULL Dataset: {}".format(data_train_pandas.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

  training_set = TitleDataset(train_dataset, tokenizer, MAX_LEN)
  val_set = TitleDataset(val_dataset, tokenizer, MAX_LEN)

  train_params = {'batch_size': TRAIN_BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0
                  }

  val_params = {'batch_size': VALID_BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0
                  }

  training_loader = DataLoader(training_set, **train_params)
  val_loader = DataLoader(val_set, **val_params)

  return training_loader, val_loader


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 19)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class TitleDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['content.title']
        self.targets = self.data['targets']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

### Train function
def train(model,
          optimizer,
          file_path,
          training_loader,
          valid_loader,
          eval_every ,
          num_epochs = 30,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    device = 'cuda' if cuda.is_available() else 'cpu'
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
      for _,data in enumerate(training_loader, 0):
       
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update running values
        running_loss += loss.item()
        global_step += 1

        # evaluation step
        if global_step % eval_every == 0:

          model.eval()
          fin_targets=[]
          fin_outputs=[]
          with torch.no_grad():
              for _, data in enumerate(valid_loader, 0):
                  ids = data['ids'].to(device, dtype = torch.long)
                  mask = data['mask'].to(device, dtype = torch.long)
                  token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                  targets = data['targets'].to(device, dtype = torch.float)
                  outputs = model(ids, mask, token_type_ids)

                  loss = loss_fn(outputs, targets)
                  valid_running_loss += loss.item()

                  fin_targets.extend(targets.cpu().detach().numpy().tolist())
                  fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

          # evaluation
          average_train_loss = running_loss / eval_every
          average_valid_loss = valid_running_loss / len(valid_loader)
          train_loss_list.append(average_train_loss)
          valid_loss_list.append(average_valid_loss)
          global_steps_list.append(global_step)

          # resetting running values
          running_loss = 0.0                
          valid_running_loss = 0.0
          model.train()

          # print progress
          print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                .format(epoch+1, num_epochs, global_step, num_epochs*len(training_loader),
                        average_train_loss, average_valid_loss))
          
          print('Mean Square Error:',mean_squared_error(fin_targets,fin_outputs))
          print('Mean Absolute Error:',mean_absolute_error(fin_targets,fin_outputs))


          # checkpoint
          if best_valid_loss > average_valid_loss:
              best_valid_loss = average_valid_loss
              save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
              save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def main():
    data_train_pandas, taxonomy_mappings = load_data()
    data_train_pandas = create_labels(data_train_pandas,taxonomy_mappings)
    
    # BERT
    device = 'cuda' if cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClass()
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    # Divide dataset
    training_loader, val_loader = get_data_loaders(data_train_pandas, tokenizer = tokenizer, percen = 0.8, debug = False)
    
    # Train
    train(model = model,
          optimizer = optimizer,
          file_path = '/content/drive/MyDrive/Ringier_Task/Bert_data_2',
          training_loader = training_loader,
          valid_loader = val_loader,
          eval_every = len(training_loader) // 2)

if __name__ == '__main__':
    main()
    