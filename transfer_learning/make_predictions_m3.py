#!/usr/bin/env python
# coding: utf-8

cd 

from fforma import *
import numpy as np
import pandas as pd
import pickle

frequencies = {'M': 7, 'Q': 4, 'Y': 1, 'Other': 1}

# Interest functions
def df_to_numpy(df):
    dict_return = {}
    for freq, df_freq in df.groupby('freq'):
        ids = df_freq['id'].unique()
        dict_return[freq] = {}
        for type_, df_type in df_freq.groupby('type'):
            list_ = [df_type.query('id==@id_')['value'].to_numpy() for id_ in ids]
            dict_return[freq][type_] = list_
        
    return dict_return

# Importing data
data_m3 = pd.read_csv('../data/data_m3/dataM3.csv')

# List train, test m3
list_m3 = df_to_numpy(data_m3)

# Training models for each frequency
models = [Naive(), SeasonalNaive(), RandomWalkDrift(), ETS()]


for freq in list_m3.keys():
    interest = list_m3[freq]
    frcy = interest['frcy']
    test_periods = interest['test_periods']
    preds = FForma().train_basic_models(models, interest['train'], frequencies[frcy]).predict_basic_models(test_periods)
    
    

