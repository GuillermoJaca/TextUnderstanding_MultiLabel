#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:00:37 2022

@author: guillermogarcia
"""
from preprocess import preprocessing
from train_title import *

class FullTextDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['content.fullTextHtml_clean']
        self.targets = self.data['targets']
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
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def get_data_loaders_full_text(data_train_pandas, tokenizer, percen = 0.8,debug = False):

  train_size = percen
  train_dataset = data_train_pandas.sample(frac=train_size,random_state=200)
  val_dataset = data_train_pandas.drop(train_dataset.index).reset_index(drop=True)
  train_dataset = train_dataset.reset_index(drop=True)

  if debug:
    print("FULL Dataset: {}".format(data_train_pandas.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

  training_set = FullTextDataset(train_dataset, tokenizer, MAX_LEN)
  val_set = FullTextDataset(val_dataset, tokenizer, MAX_LEN)

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

def main():
    data_train_pandas, taxonomy_mappings = load_data()
    data_train_pandas = create_labels(data_train_pandas,taxonomy_mappings)
    
    # Clean text    
    data_train_pandas['content.fullTextHtml_clean'] = data_train_pandas['content.fullTextHtml'].apply(lambda x: preprocessing(x))
    
    # BERT
    device = 'cuda' if cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClass()
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    # Divide dataset
    training_loader, val_loader = get_data_loaders_full_text(data_train_pandas,tokenizer, percen = 0.8, debug = False)
    
    # Train
    train(model = model,
              optimizer = optimizer,
              file_path = '/content/drive/MyDrive/Ringier_Task/Bert_fullText',
              train_loader = training_loader,
              valid_loader = val_loader,
              eval_every = len(training_loader) // 2)
    
if __name__ == '__main__':
    main()



