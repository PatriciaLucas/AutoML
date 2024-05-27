# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 08:58:44 2023

@author: Patricia
"""


import pandas as pd
import ray

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
        print(row)
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
    data = dict.fromkeys({"X_train": None, "y_train": None})
    data["X_train"], data["y_train"] = organize_dataset(dataset, G_list[target], max_lags, target)
    return data

def get_datasets_all(dataset, G_list, max_lags, distributive_version):
    dict_datasets = dict.fromkeys(list(G_list.keys()),None)
    
    
    if distributive_version:
        results = []
        for variable in dict_datasets:
            results.append(get_datasets_all_parallel.remote(dataset, G_list[variable], max_lags, variable))     
            
        parallel_results = ray.get(results)
    
        r = 0
        for variable in dict_datasets:
            data = dict.fromkeys({"X_train": None, "y_train": None})
            data["X_train"], data["y_train"] = parallel_results[r][0], parallel_results[r][1]
            dict_datasets[variable] = data
            r = r + 1
    else:
        for variable in dict_datasets:
            data = dict.fromkeys({"X_train": None, "y_train": None})
            data["X_train"], data["y_train"] = organize_dataset(dataset, G_list[variable], max_lags, variable)
            dict_datasets[variable] = data

    return dict_datasets

@ray.remote
def get_datasets_all_parallel(dataset, G_list, max_lags, variable):
    X, y = organize_dataset(dataset, G_list, max_lags, variable)
    return X, y


def load_model(path):
    import pickle

    with open(path, 'rb') as f:
        model = pickle.load(f)
        
    return model