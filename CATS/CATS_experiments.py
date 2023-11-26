# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 13:36:52 2023

@author: Patricia
"""

import sys
sys.path.append('./')
import time
import numpy as np
import pandas as pd
import cats
import measures as mea
import save_database as sd
import warnings
warnings.filterwarnings("ignore")


def execute_cats(name_dataset, dataset, target, step_ahead, database_path):
    #Criação da tabela no banco de dados:
    sd.execute("CREATE TABLE IF NOT EXISTS results(name_dataset TEXT, G BLOB, \
            time FLOAT, max_lags INT, HPO BLOB, yhats BLOB, test BLOB, nrmse FLOAT)", database_path)
    
    montecarlo_loop = 5
    train = dataset.loc[:dataset.shape[0]-301]
    test = dataset.loc[dataset.shape[0]-300:]
    
    
    start_time = time.time()
    
    #Treinamento
    G, dict_variables, dict_datasets_train, max_lags, hp_list = cats.fit(train, target)
    
    #Teste
    yhats = cats.predict(test,  step_ahead, montecarlo_loop, dict_variables, G,  max_lags, target)

    
    runtime = round(time.time() - start_time, 2)
    
    nrmse = []

    if step_ahead == 1:
        nrmse.append(mea.nrmse(test[target][max_lags:], yhats))
    else:
        for e in range(step_ahead):
            nrmse.append(mea.nrmse(test[target][max_lags+e:], yhats.loc[e,:yhats.shape[1]-(e+1)].to_frame()))
            
 
    #Salva no banco de dados
    sd.execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?)", (name_dataset, np.array(hp_list).tostring(), \
                  runtime, max_lags, np.array(hp_list).tostring(), yhats.to_numpy().tostring(), test[target].to_numpy().tostring(), 
                  np.array(nrmse).tostring()), database_path)
        
    return



def run(num_experiments=10):
  datasets = ['HOME', 'SONDA', 'DOWJONES', 'ETO', 'PRSA']
  target = ['use', 'glo_avg', 'AVG', 'ETo', 'PM2.5']
  step_ahead = 10
  database_path = 'bd_cats.db'

  for d in range(len(datasets)):
    print("Base de dados "+datasets[d])

    data = pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/CATS/datasets/'+datasets[d]+'.csv', on_bad_lines='skip')[:10000]

    for e in range(num_experiments):

      print("Experimento "+str(e))
      execute_cats(datasets[d], data, target[d], step_ahead, database_path)

  return
