# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:49:15 2023

@author: Patricia
"""

# ENSEMBLE LAYER
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from AUTODCETS import MEOHP
import random
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings



def initialize_model_layer(dict_datasets_train, target, series, params_MFEA, distributive_version, optimize_hiperparams):   

    # Dictionary that stores the ensembles of each variable in the database.
    dict_variables = dict.fromkeys(list(dict_datasets_train.keys()), {})
    if optimize_hiperparams:  
        hps, pop = MEOHP.GeneticAlgorithm(dict_datasets_train, series, params_MFEA, distributive_version)
        hp = hps[-1]
    else:
        hp = []
        for i in range(len(dict_variables)):
            hp.append(dict(n_estimators=random.randint(5, 1000), min_samples_leaf = random.randint(1, 30), max_features = random.choice(['sqrt', 'log2'])))
    
    v = 0 #variável que percorre o vetor de indivíduos retornado pelo MFEA na ordem das variáveis do dataset
    for variable in dict_variables:
        #Dictionary that stores the information of each model.
        dict_model =  {"name": hp[v]['model'], "hiperparam": hp[v], "trained_model": None, "residuals":None}
        dict_variables[variable] = dict_model
        v = v + 1
    
    return dict_variables, hps
    


def evaluate_model(dict_model, X_train, y_train):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    
    if dict_model['name'] == 'RandomForest':
        model = RandomForestRegressor(n_estimators = dict_model['hiperparam']['n_estimators'], 
                                  max_features = dict_model['hiperparam']['max_features'],
                                  min_samples_leaf = dict_model['hiperparam']['min_samples_leaf'], 
                                  bootstrap=True, n_jobs = -1)
        model.fit(X_train_copy.values, y_train_copy.values)
    elif dict_model['name'] == 'LGBoost':
        model = LGBMRegressor(n_estimators = dict_model['hiperparam']['n_estimators'], 
                                  colsample_bytree = dict_model['hiperparam']['max_features'],
                                  min_child_samples = dict_model['hiperparam']['min_samples_leaf'], 
                                  n_jobs = -1, verbosity = -1)
        model.fit(X_train_copy.values, y_train_copy.values)
    else:
        model = XGBRegressor(n_estimators = dict_model['hiperparam']['n_estimators'], 
                                  colsample_bytree = dict_model['hiperparam']['max_features'],
                                  min_child_weight = dict_model['hiperparam']['min_samples_leaf'],
                                  subsample = 0.5)
        model.fit(X_train_copy.values, y_train_copy.values)

    del X_train
    del y_train
    
    forecasts = model.predict(X_train_copy.values)
    max_lags = y_train_copy.shape[0] - forecasts.shape[0]
    residuals = y_train_copy[max_lags:].values - forecasts
    return model, residuals



