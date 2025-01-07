# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:29:49 2024

@author: Patricia
"""

from MARTS import save_database as sd
from MARTS import marts
from MARTS import measures
from MARTS import util
import pandas as pd


def experiments(dataset_all, target, name_dataset, type_experiment, window_n, database_path, j, path_model, step_ahead):
    
    sd.execute("CREATE TABLE IF NOT EXISTS results(type_experiment TEXT, name_dataset TEXT, step_ahead INT, num_variables INT, \
               rmse FLOAT, mape FLOAT, nrmse FLOAT, theil FLOAT)", database_path)
    
    w = int(len(dataset_all)/window_n)

    for i in range(j, window_n):
        
        dataset = dataset_all[i*w:(i*w)+w]
        
        params_MFEA = {
            'npop': 20,
            'ngen': 10,
            'mgen': 5,
            'psel': 0.5,
            'size_train': (dataset.shape[0] - (dataset.shape[0]*0.2))/5,
            'size_test': (dataset.shape[0]*0.2)/5,
            }
        path_model = path_model+'_'+str(i)

        #runs the full marts only in the first window
        if i >= 0:
            if type_experiment == 'A': #Sem seleção e sem decomposição
                model = marts.Marts(params_MFEA = params_MFEA, feature_selection = False, distributive_version = True, 
                                save_model = True, path_model = path_model, decomposition = False, test_size=dataset.shape[0]*0.2, size_dataset_optimize_max_lags=3,
                                optimize_hiperparams = True)
                model.fit(dataset, target)
                model = util.load_model(path_model+'.pickle')
                df_results = model.predict(step_ahead=step_ahead)
            elif type_experiment == 'B': #Com seleção e sem decomposição
                model = marts.Marts(params_MFEA = params_MFEA, feature_selection = True, distributive_version = True, 
                                save_model = True, path_model = path_model, decomposition = False, test_size=dataset.shape[0]*0.2, size_dataset_optimize_max_lags=3,
                                optimize_hiperparams = True)
                model.fit(dataset, target)
                df_results = model.predict(step_ahead=step_ahead)
            elif type_experiment == 'C': #Com seleção e com decomposição
                model = marts.Marts(params_MFEA = params_MFEA, feature_selection = True, distributive_version = True, 
                                save_model = True, path_model = path_model, decomposition = True, test_size=dataset.shape[0]*0.2, size_dataset_optimize_max_lags=3,
                                optimize_hiperparams = True)
                model.fit(dataset, target)
                df_results = model.predict_decom(step_ahead=step_ahead)
            elif type_experiment == 'D': #Com seleção e com decomposição - problemas univariados
                model = marts.Marts(params_MFEA = params_MFEA, feature_selection = True, distributive_version = True, 
                                save_model = True, path_model = path_model, decomposition = True, test_size=dataset.shape[0]*0.2, size_dataset_optimize_max_lags=3,
                                optimize_hiperparams = True)
                model.fit(dataset, target)
                df_results = model.predict_decom(step_ahead=step_ahead)
            
        else:
            
            model = util.load_model(path_model+'.pickle')
            model.retraining(dataset)
            if type_experiment == 'A' or type_experiment == 'B':
                df_results = model.predict(step_ahead=step_ahead)
            else:
                df_results = model.predict_decom(step_ahead=step_ahead)

        #calculates the metrics
        mea = measures.Measures(model)
        if type_experiment == 'C' or type_experiment == 'D':
            results = mea.score(pd.DataFrame(model.target_test, columns=[model.target]), df_results)
        else:
            results = mea.score(model.test, df_results)
        
        print(results)
        
        for r in range(step_ahead):
            sd.execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?, ?)", (type_experiment, name_dataset, r, model.num_variables, \
                           results['rmse'][r], results['mape'][r], results['nrmse'][r], results['theil'][r],
                           ), database_path)
             
        print("Save: ",name_dataset)
        print("Experiment: ",type_experiment)