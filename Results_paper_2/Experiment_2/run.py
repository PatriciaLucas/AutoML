# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:29:49 2024

@author: Patricia
"""


import pandas as pd
import time
import numpy as np
from AUTOTSF import save_database as sd
import models
import datetime




def experiments(dataset_all, target, name_dataset, name_model, database_path, janela):
    
    sd.execute("CREATE TABLE IF NOT EXISTS results(name_dataset TEXT, name_model TEXT, step_ahead INT, window INT, time FLOAT, \
               real FLOAT, forecast FLOAT)", database_path) #8 colunas
    
    step_ahead = 30
    windows_size = .3
    test_size = 30
    w = int(dataset_all.shape[0] * windows_size)
    d = int(0.1 * w)
    i=janela

    while i*d < w:
        print(f"--------------------> JANELA {i}")
        dataset = dataset_all[i*d:(i*d)+w]
        #dataset = dataset.interpolate()
        #train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size)
        
        if name_model == 'PROPOSTO':

            start_time = time.time()
            
            real, forecast = models.proposto(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
        
        elif name_model == 'AUTOGLUON':
            
            start_time = time.time()
            print(f"Início: {datetime.datetime.now()}")
            
            real, forecast = models.autogluon(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
        
        elif name_model == 'FEDOT':
            print(f"Início: {datetime.datetime.now()}")
            start_time = time.time()
            
            real, forecast = models.fedot(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
            
        elif name_model == 'PYCARET':
            start_time = time.time()
            print(f"Início: {datetime.datetime.now()}")

            real, forecast = models.pycaret(dataset, test_size, target)

            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
            
        elif name_model == 'AUTOTS': 
            start_time = time.time()
            print(f"Início: {datetime.datetime.now()}")
            
            real, forecast = models.autots(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
        
        elif name_model == 'LIGHTGBM': 
            start_time = time.time()
            print(f"Início: {datetime.datetime.now()}")
            
            real, forecast = models.LIGHTGBM(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
            
        elif name_model == 'RF': 
            start_time = time.time()
            print(f"Início: {datetime.datetime.now()}")
            
            real, forecast = models.RF(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
            
        elif name_model == 'XGBOOST': 
            start_time = time.time()
            print(f"Início: {datetime.datetime.now()}")
            
            real, forecast = models.XGBOOST(dataset, test_size, target)
            
            runtime = round(time.time() - start_time, 2)
            print(f"Fim: {datetime.datetime.now()}")
            

        for r in range(step_ahead):
                sd.execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?)", (name_dataset, \
                                name_model, r, i, runtime, 
                                real[r], forecast[r]), database_path)
         
        i = i+1
        print("Save: ",name_dataset)
        print("Model: ",name_model)

           
 


