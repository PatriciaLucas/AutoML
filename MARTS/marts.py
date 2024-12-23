# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:17:53 2023

@author: Patricia
"""


from MARTS import feature_selection as fs
from MARTS import model_generation as mg
from MARTS import util
from MARTS import forecast as fo

import pandas as pd
import numpy as np
import ray
import os
import random
import pickle
import emd
from sklearn.neighbors import KernelDensity
import datetime




class Marts():
    def __init__(self, 
                 params_MFEA = {'npop': 20,'ngen': 10,'mgen': 5,'psel': 0.5,'size_train': 100,'size_test': 50,}, 
                 feature_selection = True, 
                 distributive_version = True,
                 save_model = True,
                 path_model = 'model',
                 decomposition = True,
                 max_lags = 5,
                 test_size = 100,
                 size_dataset_optimize_max_lags = 3,
                 optimize_hiperparams = True
                 ):
        self.params_MFEA = params_MFEA
        self.feature_selection = feature_selection
        self.distributive_version = distributive_version
        self.save_model = save_model
        self.path_model = path_model
        self.decomposition = decomposition
        self.G_list = {}
        self.dict_variables = {}
        self.dict_datasets_train = {}
        self.dict_datasets_test = {}
        self.max_lags = max_lags
        self.target = ''
        self.target_test = []
        self.imfs = []
        self.test = []
        self.test_size = test_size
        self.size_dataset_optimize_max_lags = size_dataset_optimize_max_lags
        self.optimize_hiperparams = optimize_hiperparams
        self.pca = []
        self.hp = []
        self.num_variables = 0


    def fit(self, dataset, target):
        
        print(f"Start time: {datetime.datetime.now()}")
        
        dataset.index = range(0,dataset.shape[0])
        
        if self.distributive_version:
            num_cpu = os.cpu_count()
            
            if not ray.is_initialized():
                context = ray.init(num_cpus=num_cpu)
                print(context.dashboard_url)
                
        self.target = target
        
        #Exclui séries constantes
        for variable in dataset.columns:
            if dataset[variable].max() == dataset[variable].min():
                dataset = dataset.drop(variable, axis=1)
                print(f"Variables {variable} were deleted because they are constant.")
                
        
        
        #Empirical Mode Decomposition
        print('FEATURE EXTRACTION LAYER')
        if self.decomposition:
            imf = emd.sift.sift(dataset[self.target].values)
            self.imfs = pd.DataFrame(imf, columns=(["IMF"+str(i) for i in range(1,imf.shape[1]+1)]))
            dataset = pd.concat([dataset,self.imfs], axis=1)
            train = dataset.loc[:dataset.shape[0]-self.test_size+1]
        else:
            train = dataset.loc[:dataset.shape[0]-self.test_size+1]
            

        # FEATURE SELECTION LAYER
        print('FEATURE SELECTION LAYER')
        #try:
         #   self.max_lags = fs.optimize_max_lags(train.loc[:train.shape[0]/self.size_dataset_optimize_max_lags], self.target)
        #except:
            # Em caso de séries IMFs constantes
        self.max_lags = 20
        print(f"Lag window size: {self.max_lags}")
        
        #Separa os dados de teste de acordo com os lags
        self.test = dataset.loc[dataset.shape[0]-self.test_size-self.max_lags:]
        
        if self.test_size != 0:
          self.target_test = dataset.loc[dataset.shape[0]-self.test_size:][self.target]
          self.target_test.index = range(0,self.target_test.shape[0])
        
        
        if self.feature_selection:
            if self.decomposition:
                train = train.drop(self.target, axis=1)
                self.test = self.test.drop(self.target, axis=1)
                self.G_list = fs.causal_graph(train.loc[:train.shape[0]], "", self.max_lags)
            else:
                self.G_list = fs.causal_graph(train.loc[:train.shape[0]], self.target, self.max_lags)
            print("Causal graph of variables")
            print(self.G_list.keys())
        else:
            print("Gera grafo completo")
            self.G_list = fs.complete_graph(train.loc[:train.shape[0]/2], self.target, self.max_lags)         
        
        
        # DELETA VARIÁVEIS QUE NÃO ESTÃO NO GRAFO CAUSAL
        variable_delete = []
        train_columns_values = set(list(train.columns.values))
        keys_G_list = set(self.G_list)
        variable_delete  = train_columns_values - keys_G_list
        train = train.drop(variable_delete, axis=1)
        
        variable_delete = []
        test_columns_values = set(list(self.test.columns.values))
        keys_G_list = set(self.G_list)
        variable_delete  = test_columns_values - keys_G_list
        self.test = self.test.drop(variable_delete, axis=1)
        self.test.index = range(0,self.test.shape[0])
        
        
        if variable_delete:
            print(f"Variables {variable_delete} were deleted because they did not have predictive lags.")
            
            
        # #DENOISING TRAINING TECHNIQUE (seleciona 50% das amostras de uma coluna e insere o ruído na amostra)
        # for variable in train.columns:
        #     index_rows = train[variable].sample(frac=0.5, random_state=42).index
        #     train.loc[index_rows, variable] = train.loc[index_rows, variable] + random.random() * np.random.normal(loc=0, scale=1) * train[variable].std()
        
            
    
        # MODEL SELECTION LAYER
        print('MODEL SELECTION LAYER')
        self.dict_datasets_train = util.get_datasets_all(train, self.G_list, self.max_lags, self.distributive_version)
        
        try:
            self.num_variables = self.dict_datasets_train[self.target]['X_train'].shape[1]
        except:
            self.num_variables = 0
        
        self.dict_variables, self.hp = mg.initialize_model_layer(self.dict_datasets_train, self.target, train, self.params_MFEA, self.distributive_version, self.optimize_hiperparams)

        for variable in self.dict_datasets_train:
            self.dict_variables[variable]["trained_model"], self.dict_variables[variable]["residuals"] = mg.evaluate_model(self.dict_variables[variable], self.dict_datasets_train[variable]['X_train'], self.dict_datasets_train[variable]['y_train'])


        if self.save_model:
            with open(self.path_model+".pickle", 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            
            
        if self.distributive_version:
            if ray.is_initialized():
                ray.shutdown()  
        
    
    # ENDOGENOUS PREDICTION LAYER
    def predict_ahead(self, step_ahead):
            print("MODEL PREDICTING")
            test = self.test
        
            if self.distributive_version:
                num_cpu = os.cpu_count()
                
                if not ray.is_initialized():
                    ray.init(num_cpus=num_cpu)
                
            df_results = pd.DataFrame()
        
            #test.index = range(0,test.shape[0])
            if self.test_size != 0:
                test_minus_max_lags = test.shape[0] - self.max_lags
            else:
                test_minus_max_lags = 1
            for row in range(test_minus_max_lags):
                
                block = test.loc[row:row + self.max_lags - 1]
                block.index = range(0,block.shape[0])
                
                ### EXOGENOUS PREDICTION LAYER
                block_forecast = fo.exogenous_forecast(step_ahead, block, self.max_lags, self.dict_variables, self.G_list)
                
                imfs = block_forecast.filter(regex='IMF')
                
                df_results[row] = imfs.sum(axis=1).values
           
            if self.distributive_version:
                if ray.is_initialized():
                    ray.shutdown() 

            return df_results
        
        
        
    def predict_ahead_mult(self, step_ahead, target):
            print("MODEL PREDICTING")
            test = self.test
        
            if self.distributive_version:
                num_cpu = os.cpu_count()
                
                if not ray.is_initialized():
                    ray.init(num_cpus=num_cpu)
                
            df_results = pd.DataFrame()
        
            #test.index = range(0,test.shape[0])
            if self.test_size != 0:
                test_minus_max_lags = test.shape[0] - self.max_lags
            else:
                test_minus_max_lags = 1
            for row in range(test_minus_max_lags):
                block = test.loc[row:row + self.max_lags - 1]
                block.index = range(0,block.shape[0])
                
                ### EXOGENOUS PREDICTION LAYER
                block_forecast = fo.exogenous_forecast(step_ahead, block, self.max_lags, self.dict_variables, self.G_list)
                
                df_results = block_forecast
           
            if self.distributive_version:
                if ray.is_initialized():
                    ray.shutdown() 

            return df_results
        
    
   

    

              
        