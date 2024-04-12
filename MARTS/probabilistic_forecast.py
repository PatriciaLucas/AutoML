# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:53:57 2023

@author: Patricia
"""

import util
import numpy as np
from sklearn.neighbors import KernelDensity


def montecarlo_forecast(step_ahead, block, max_lags, bootstrap_size, target, dict_variables, G_list):
    
    #Gera as distribuições dos residuos
    kde = dict.fromkeys(list(block.columns.values),None)
    for variable in block.columns.values.tolist():
        kde[variable] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dict_variables[variable][0]['residuals'].reshape((-1,1)))

    
    for step in range(step_ahead-2):
        first_forecast = []
        array_bootstrap = []
        
        if step==0:
            for variable in block.columns.values.tolist():
                model = dict_variables[variable][0]['trained_model']
                X_input = util.organize_block(block, G_list[variable], max_lags)
                forecast = model.predict(X_input)[0]
                first_forecast.append(forecast)
        
        if step != 0:
            for k in range(bootstrap_size):
                array_bootstrap.append(montecarlo(block, dict_variables, G_list, max_lags, kde))
                      
        
            p_bootstrap = []
            for v in range(block.shape[1]):
                kde_2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np.array(array_bootstrap)[:,v].reshape((-1,1)))
                p_bootstrap.append((kde_2.sample(100)).mean())
        
        if step == 0:
            block.loc[max_lags] = first_forecast
            block = block.drop([0])
            block.index = range(0,block.shape[0])
        else:
            block.loc[max_lags] = p_bootstrap
            block = block.drop([0])
    
    block = block[block.shape[0] - (step_ahead - 1):]
    block.index = range(0,block.shape[0])
    
    
    return block


def montecarlo(block, dict_variables, G_list, max_lags, kde):
    
    p = []
    
    for variable in block.columns.values.tolist():
        residual = (kde[variable].sample(100)).mean()
        block[variable][max_lags] = block[variable][max_lags] + residual
    
    for variable in block.columns.values.tolist():
        model = dict_variables[variable][0]['trained_model']
        X_input = util.organize_block(block, G_list[variable], max_lags)
        forecast = model.predict(X_input)[0]
        
        residual = (kde[variable].sample(100)).mean()
        p.append(residual + forecast)
        
        
    return p