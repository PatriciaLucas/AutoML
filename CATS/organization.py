# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 08:58:44 2023

@author: Patricia
"""


import pandas as pd

def organize_dataset(dataset, G, max_lags, target):
    dataset.index = range(0,dataset.shape[0])
    lags = G.where(G).stack().index.tolist()
    y = dataset[target].loc[max_lags:]
    y.index = range(0,y.shape[0])
    try:
        for row in range(0,y.shape[0]):
            cols = []
            values = []
            bloco = dataset.iloc[row:max_lags+row]   
            bloco.index = reversed(range(1,bloco.shape[0]+1))
            for lag in lags:
                if row == 0:
                    cols.append(lag[1])
                    X = pd.DataFrame(columns=cols)
                values.append(bloco[lag[1]].loc[lag[0]])
    
            X.loc[row,:] = values
    except:
        print("O PCMCI não encontrou links causais para a variável "+target)
        print("Aumente o número de lags observados.")

    return X, y

def organize_block(dataset, G, max_lags):
    dataset.index = range(0,dataset.shape[0])
    lags = G.where(G).stack().index.tolist()

    for row in range(0,1):
        cols = []
        values = []
        bloco = dataset.iloc[row:max_lags+row]   
        bloco.index = reversed(range(1,bloco.shape[0]+1))
        for lag in lags:
            if row == 0:
                cols.append(lag[1])
                X = pd.DataFrame(columns=cols)
            values.append(bloco[lag[1]].loc[lag[0]])
        
        X.loc[row,:] = values
    return X


def get_datasets(dataset, G_list, max_lags, target):
    data = dict.fromkeys({"X": None, "y": None})
    data["X"], data["y"] = organize_dataset(dataset, G_list[target], max_lags, target)
    return data

def get_datasets_all(dataset, G_list, max_lags):
    dict_datasets = dict.fromkeys(list(G_list.keys()),None)
    for variable in dict_datasets:
        data = dict.fromkeys({"X": None, "y": None})
        data["X"], data["y"] = organize_dataset(dataset, G_list[variable], max_lags, variable)
        dict_datasets[variable] = data
    return dict_datasets

