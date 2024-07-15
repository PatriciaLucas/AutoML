# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:53:57 2023

@author: Patricia
"""
import sys
sys.path.append('./')
import util
import ray
import random



def exogenous_forecast(step_ahead, block, max_lags, dict_variables, G_list):

    for step in range(step_ahead):
            
        p = []
        
        for variable in block.columns.values.tolist():
            p.append(until_organize_block.remote(block, G_list, max_lags, variable, dict_variables))
        
        parallel_p = ray.get(p)
        block.loc[max_lags+1] = parallel_p
        block.index = range(0,block.shape[0])
        block = block.drop([0])
        block.index = range(0,block.shape[0])
            
    
    block = block[block.shape[0] - (step_ahead):]
    block.index = range(0,block.shape[0])
    
    return block

@ray.remote
def until_organize_block(block, G_list, max_lags, variable, dict_variables):
    model = dict_variables[variable]['trained_model']
    X_input = util.organize_block(block, G_list[variable], max_lags)
    forecast = model.predict(X_input.values)[0]
    residual = random.choice(dict_variables[variable]['residuals'])
    return residual + forecast




        
        
        