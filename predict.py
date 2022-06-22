#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:56:37 2022

@author: guillermogarcia
"""
import json
import pandas as pd
import numpy as np
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
import argparse

from preprocess import preprocessing
from train_title import BERTClass

class FullTextDatasetPredict(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['content.fullTextHtml_clean']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

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
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']



def load_data(path):
    with open(path, 'r') as f:
        data_predict_payload_json = json.load(f)

    data_predict_payload_pandas = pd.json_normalize(data_predict_payload_json)
    return data_predict_payload_pandas


def get_fulltext_data_to_predict(predict_dataset):
  
    #Preprocessing
    predict_dataset['content.fullTextHtml_clean'] = predict_dataset['content.fullTextHtml'].apply(lambda x: preprocessing(x))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LEN = 512
    PREDICT_BATCH_SIZE = 1
    predict_set = FullTextDatasetPredict(predict_dataset, tokenizer, MAX_LEN)
    
    predict_params = {'batch_size': PREDICT_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }
    
    predict_loader = DataLoader(predict_set, **predict_params)
    
    return predict_loader


def predict(model,valid_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.eval()
  fin_outputs=[]
  with torch.no_grad():
      for _, data in enumerate(valid_loader, 0):
          ids = data['ids'].to(device, dtype = torch.long)
          mask = data['mask'].to(device, dtype = torch.long)
          token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)

          outputs = model(ids, mask, token_type_ids)
          fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

  return fin_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--data_path', '-dp', type=str, default='predict_payload.json',
                        help='path to data to predict')
    args = parser.parse_args()
    
    # Data to Predict
    predict_dataset = load_data(args.data_path)
    predict_loader = get_fulltext_data_to_predict(predict_dataset)
    
    # Model
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = BERTClass().to(device)
    load_checkpoint('model.pt', model)
    
    #predict
    print(predict(model = model,valid_loader = predict_loader))








