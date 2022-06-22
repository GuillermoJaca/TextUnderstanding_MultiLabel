#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:56:38 2022

@author: guillermogarcia
"""

import pandas as pd
import numpy as np
import re

def preprocessing(x):
  text = re.sub('\<.*?\>','', x)
  text = text.replace("\"", "")
  return " ".join(text.split())

def func(x,inverse_taxonomy_mappings):
  aux_list = np.zeros(len(inverse_taxonomy_mappings))
  for result in x:
    num = inverse_taxonomy_mappings[result[0]]
    aux_list[int(num)] = result[1]

  return aux_list

def create_labels(data_train_pandas,taxonomy_mappings):
  #Create the inverse dictionary of taxonomy_mapping
  inv_map = {v: k for k, v in taxonomy_mappings.items()}
  #Create a list for the target probability of each sample
  data_train_pandas['targets'] = data_train_pandas['labels'].apply(lambda x: func(x,inv_map))

  return data_train_pandas











