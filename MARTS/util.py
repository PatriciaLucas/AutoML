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
    distributive_version = False
    # try:
    if distributive_version:
        y_shape = y.shape[0]
        results = []
        for row in range(0,y_shape):
            results.append(get_values_lags.remote(dataset, row, max_lags, lags))
        
        parallel_results = ray.get(results)
        X = pd.DataFrame(parallel_results, columns =[l[1] for l in lags])
    else:
        y_shape = y.shape[0]
        values = []
        for row in range(0,y_shape):
            bloco = dataset.iloc[row:max_lags+row]   
            bloco.index = reversed(range(1,bloco.shape[0]+1))
            values.append([bloco.loc[lag[0], lag[1]] for lag in lags])

        X = pd.DataFrame(values, columns =[l[1] for l in lags])
    # except:
    #     print("O PCMCI não encontrou links causais para a variável "+target)
    #     print("Aumente o número de lags observados.")

    return X, y

@ray.remote
def get_values_lags(dataset, row, max_lags, lags):
    bloco = dataset.iloc[row:max_lags+row]   
    bloco.index = reversed(range(1,bloco.shape[0]+1))
    values = [bloco.loc[lag[0], lag[1]] for lag in lags]
    # for lag in lags:
    #     values.append(bloco[lag[1]].loc[lag[0]])

    return values


def organize_block(dataset, G, max_lags):
    dataset.index = range(0,dataset.shape[0])
    lags = G.where(G).stack().index.tolist()
    bloco = dataset.iloc[0:max_lags+0]   
    bloco.index = reversed(range(1,bloco.shape[0]+1))
    X = pd.DataFrame(columns=[l[1] for l in lags])

    X.loc[0,:] = [bloco.loc[lag[0], lag[1]] for lag in lags]

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