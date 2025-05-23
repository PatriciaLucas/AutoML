# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:53:57 2023

@author: Patricia
"""

from AUTODCETS import util

import ray
import random
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



def exogenous_forecast(step_ahead, block, max_lags, dict_variables, G_list, distributive_version):
    
    block_forecast = pd.DataFrame(columns=block.columns.values)

    for step in range(step_ahead):
  
        p = []
        
        if distributive_version:
            for variable in block.columns.values.tolist():
                p.append(until_organize_block.remote(block, G_list, max_lags, variable, dict_variables))
            parallel_p = ray.get(p)
            block = block.copy()
            block.loc[block.shape[0]] = parallel_p
            block_forecast = block_forecast.copy()
            block_forecast.loc[block_forecast.shape[0]] = parallel_p
            
        else:
            p = []
            for variable in block.columns.values.tolist():
                model = dict_variables[variable]['trained_model']
                X_input = util.organize_block(block, G_list[variable], max_lags)
                forecast = model.predict(X_input.values)[0]
                residual = np.mean(dict_variables[variable]['residuals'])
                #p.append(forecast + residual)
                p.append(forecast)
            block = block.copy()
            block.loc[block.shape[0]] = p
            block_forecast = block_forecast.copy()
            block_forecast.loc[block_forecast.shape[0]] = p
            

        #parallel_p = ray.get(p)
        #block.loc[block.shape[0]] = parallel_p
        #block_forecast.loc[block_forecast.shape[0]] = parallel_p
        block = block.drop([0])
        block.index = range(0,block.shape[0])
            
    
    return block_forecast

@ray.remote
def until_organize_block(block, G_list, max_lags, variable, dict_variables):
    model = dict_variables[variable]['trained_model']
    X_input = util.organize_block(block, G_list[variable], max_lags)
    forecast = model.predict(X_input.values)[0]
    residual = np.mean(dict_variables[variable]['residuals'])
    #return forecast + residual
    return forecast



def exogenous_forecast_prob(step_ahead, block, max_lags, dict_variables, G_list):
    
    block_forecast = pd.DataFrame(columns=block.columns.values)

    for step in range(step_ahead):
  
        p = []
        
        for variable in block.columns.values.tolist():
            p.append(until_organize_block_prob.remote(block, G_list, max_lags, variable, dict_variables))

        parallel_p = ray.get(p)
        block.loc[block.shape[0]] = parallel_p
        block_forecast.loc[block_forecast.shape[0]] = parallel_p
        block = block.drop([0])
        block.index = range(0,block.shape[0])
            
    
    #block = block[block.shape[0] - (max_lags):]
    #block.index = range(0,block.shape[0])
    
    return block_forecast
        
@ray.remote
def until_organize_block_prob(block, G_list, max_lags, variable, dict_variables):
    model = dict_variables[variable]['trained_model']
    X_input = util.organize_block(block, G_list[variable], max_lags)
    forecast = model.predict(X_input.values)[0]
    residual = np.mean(dict_variables[variable]['residuals'])
    return forecast + residual        
        