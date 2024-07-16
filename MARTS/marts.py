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
        if self.decomposition:
            imf = emd.sift.sift(dataset[self.target].values)
            self.imfs = pd.DataFrame(imf, columns=(["IMF"+str(i) for i in range(1,imf.shape[1]+1)]))
            dataset = pd.concat([dataset,self.imfs], axis=1)
                    
            train = dataset.loc[:dataset.shape[0]-self.test_size+1]
            self.test = dataset.loc[dataset.shape[0]-self.test_size:]
            self.target_test = self.test[self.target]
        else:
            train = dataset.loc[:dataset.shape[0]-self.test_size+1]
            self.test = dataset.loc[dataset.shape[0]-self.test_size:]
            
        print(train.loc[:train.shape[0]/self.size_dataset_optimize_max_lags])
        # FEATURE SELECTION LAYER
        self.max_lags = fs.optimize_max_lags(train.loc[:train.shape[0]/self.size_dataset_optimize_max_lags], self.target)
        #self.max_lags = 5
        print(f"Number of lags: {self.max_lags}")
        
        
        if self.feature_selection:
            if self.decomposition:
                train = train.drop(self.target, axis=1)
                self.test = self.test.drop(self.target, axis=1)
                self.G_list = fs.causal_graph(train, "", self.max_lags)
            else:
                self.G_list = fs.causal_graph(train, self.target, self.max_lags)
            print("Causal graph of variables")
            print(self.G_list.keys())
        else:
            print("Gera grafo completo")
            self.G_list = fs.complete_graph(train, self.target, self.max_lags)        
                
        variable_delete = []
        train_columns_values = set(list(train.columns.values))
        keys_G_list = set(self.G_list)
        variable_delete  = train_columns_values - keys_G_list
        train = train.drop(variable_delete, axis=1)
        
        
        if variable_delete:
            print(f"Variables {variable_delete} were deleted because they did not have predictive lags.")
            
    
        # MODEL SELECTION LAYER
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
    def predict(self, step_ahead):
            test = self.test
            test.index = range(0,test.shape[0])
        
            if self.distributive_version:
                num_cpu = os.cpu_count()
                
                if not ray.is_initialized():
                    ray.init(num_cpus=num_cpu)
            

            variable_delete = []
            test_columns_values = set(list(test.columns.values))
            keys_G_list = set(self.G_list)
            variable_delete  = test_columns_values - keys_G_list
            test = test.drop(variable_delete, axis=1)
        
        
            model = self.dict_variables[self.target]['trained_model']
            residual = self.dict_variables[self.target]['residuals']
                
            df_results = pd.DataFrame()   
        
            test.index = range(0,test.shape[0])
            test_minus_max_lags = test.shape[0] - self.max_lags
            for row in range(test_minus_max_lags):
                
                block_all = test.loc[row:row + self.max_lags]
                
                ### EXOGENOUS PREDICTION LAYER
                if step_ahead > 1:
                    block = block_all.drop([self.target], axis=1)
                    block_forecast = fo.exogenous_forecast(step_ahead-1, block, self.max_lags, self.dict_variables, self.G_list)
                    block_nan = pd.concat([block_all, block_forecast])
                else:
                    block_nan = block_all
                ###
                
                block_nan.index = range(0,block_nan.shape[0])
                block_nan = block_nan.drop([0])
                block_nan.index = range(0,block_nan.shape[0])
                
                for f in range(0,step_ahead):
                    X = util.organize_block(block_nan.iloc[f:], self.G_list[self.target], self.max_lags)
    
                    if X.isnull().any().any():
                        break
                    else:
                        residual = random.choice(self.dict_variables[self.target]['residuals'])
                        forecast = model.predict(X.values) + residual
                    
                    try:
                        block_nan[self.target].iloc[self.max_lags + f] = forecast
                    except:
                        df = pd.DataFrame(np.full([1, block_nan.shape[1],], np.nan), columns = block_nan.columns.values)
                        block_nan = pd.concat([block_nan, df])
                        block_nan[self.target].iloc[self.max_lags + f] = forecast
                        

                block_nan.index = range(0,block_nan.shape[0])
                df_results[row] = block_nan[self.target].iloc[block_nan.shape[0]-step_ahead:].values
           
            if self.distributive_version:
                if ray.is_initialized():
                    ray.shutdown() 

            return df_results
    
    # ENDOGENOUS PREDICTION LAYER
    def predict_decom(self, step_ahead):
            test = self.test
            test.index = range(0,test.shape[0])
        
            if self.distributive_version:
                num_cpu = os.cpu_count()
                
                if not ray.is_initialized():
                    ray.init(num_cpus=num_cpu)
            

            variable_delete = []
            test_columns_values = set(list(test.columns.values))
            keys_G_list = set(self.G_list)
            variable_delete  = test_columns_values - keys_G_list
            test = test.drop(variable_delete, axis=1)
        
        
            #model = self.dict_variables[self.target]['trained_model']
            #residual = self.dict_variables[self.target]['residuals']
                
            df_results = pd.DataFrame()
        
            test.index = range(0,test.shape[0])
            test_minus_max_lags = test.shape[0] - self.max_lags
            for row in range(test_minus_max_lags):
                
                block = test.loc[row:row + self.max_lags]
                
                ### EXOGENOUS PREDICTION LAYER
                block_forecast = fo.exogenous_forecast(step_ahead, block, self.max_lags, self.dict_variables, self.G_list)
                
                imfs = block_forecast.filter(regex='IMF')
                
                df_results[row] = imfs.sum(axis=1).values
           
            if self.distributive_version:
                if ray.is_initialized():
                    ray.shutdown() 

            return df_results
    
    def predict_prob(self, step_ahead, prob_forecast=False):
            test = self.test
            test.index = range(0,test.shape[0])
        
            if self.distributive_version:
                num_cpu = os.cpu_count()
                
                if not ray.is_initialized():
                    ray.init(num_cpus=num_cpu)
            

            variable_delete = []
            test_columns_values = set(list(test.columns.values))
            keys_G_list = set(self.G_list)
            variable_delete  = test_columns_values - keys_G_list
            test = test.drop(variable_delete, axis=1)
        
        
            model = self.dict_variables[self.target]['trained_model']
            residual = self.dict_variables[self.target]['residuals']
                
            df_results = pd.DataFrame()
            df_distributions = pd.DataFrame()    
        
            test.index = range(0,test.shape[0])
            test_minus_max_lags = test.shape[0] - self.max_lags
            for row in range(test_minus_max_lags):
                
                block_all = test.loc[row:row + self.max_lags]
                
                ### EXOGENOUS PREDICTION LAYER
                if step_ahead > 1:
                    if block_all.shape[1] != 1:
                        block = block_all.drop([self.target], axis=1)
                    else:
                        block = block_all
                    block_forecast = fo.exogenous_forecast(step_ahead-1, block, self.max_lags, self.dict_variables, self.G_list)
                    block_nan = pd.concat([block_all, block_forecast])
                else:
                    block_nan = block_all
                ###
                
                
                block_nan.index = range(0,block_nan.shape[0])
                block_nan = block_nan.drop([0])
                block_nan.index = range(0,block_nan.shape[0])
                
                distributions = []
                for f in range(0,step_ahead):
                    X = util.organize_block(block_nan.iloc[f:], self.G_list[self.target], self.max_lags)
    
                    if X.isnull().any().any():
                        break
                    else:
                        if prob_forecast:
                            predictions_tree = np.array([tree.predict(X) for tree in model.estimators_])
                            kde_predictions_tree = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(predictions_tree.reshape((-1,1)))
                        else:
                            residual = random.choice(self.dict_variables[self.target]['residuals'])
                            forecast = model.predict(X.values) + residual
                    
                    if prob_forecast:
                        distributions.append(kde_predictions_tree)
                    else:
                        try:
                            block_nan[self.target].iloc[self.max_lags + f] = forecast
                                
                        except:
                            df = pd.DataFrame(np.full([1, block_nan.shape[1],], np.nan), columns = block_nan.columns.values)
                            block_nan = pd.concat([block_nan, df])
                            block_nan[self.target].iloc[self.max_lags + f] = forecast
                        
                
                if prob_forecast:
                    df_distributions[row] = distributions
                else:
                    block_nan.index = range(0,block_nan.shape[0])
                    df_results[row] = block_nan[self.target].iloc[block_nan.shape[0]-step_ahead:].values
           
            if self.distributive_version:
                if ray.is_initialized():
                    ray.shutdown() 
            if prob_forecast:       
                return df_distributions
            else:
                return df_results

    def retraining(self, dataset):
        
        dataset.index = range(0,dataset.shape[0])
        
        if self.distributive_version:
            num_cpu = os.cpu_count()
            
            if not ray.is_initialized():
                ray.init(num_cpus=num_cpu)
        
        #Empirical Mode Decomposition
        if self.decomposition:
            imf = emd.sift.sift(dataset[self.target].values, max_imfs = self.imfs.shape[1])
            self.imfs = pd.DataFrame(imf, columns=(["IMF"+str(i) for i in range(1,imf.shape[1]+1)]))
            dataset = pd.concat([dataset,self.imfs], axis=1)

        
        variable_delete = []
        dataset_columns_values = set(list(dataset.columns.values))
        keys_G_list = set(self.G_list)
        variable_delete  = dataset_columns_values - keys_G_list
        dataset = dataset.drop(variable_delete, axis=1)
        variable_delete  = keys_G_list - dataset_columns_values
        print(variable_delete)
        # for k in variable_delete:
        #     del self.G_list[k]
        # for var in self.G_list:
        #     for k in variable_delete:
        #         if "IMF" in var:
        #             del self.G_list[var][k]

        
        if self.decomposition:
            train = dataset.loc[:dataset.shape[0]-self.test_size+1]
            #train = train.drop(self.target, axis=1)
            self.test = dataset.loc[dataset.shape[0]-self.test_size:]
        else:
            train = dataset.iloc[:dataset.shape[0]-self.test_size+1,:]
            self.test = dataset.iloc[dataset.shape[0]-self.test_size:,:]
            
        # MODEL SELECTION LAYER
        self.dict_datasets_train = util.get_datasets_all(train, self.G_list, self.max_lags, self.distributive_version)
    
        for variable in self.dict_datasets_train:
            self.dict_variables[variable]["trained_model"], self.dict_variables[variable]["residuals"] = mg.evaluate_model(self.dict_variables[variable], self.dict_datasets_train[variable]['X_train'], self.dict_datasets_train[variable]['y_train'])
            
        # if self.save_model:
        #     with open('model.pickle', 'wb') as f:
        #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                
        if self.distributive_version:
            if ray.is_initialized():
                ray.shutdown() 

              
        