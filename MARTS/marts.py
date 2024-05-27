# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:17:53 2023

@author: Patricia
"""
import feature_selection as fs
import model_generation as mg
import util
import probabilistic_forecast as pf
import pandas as pd
import numpy as np
import ray
import os
from sklearn.neighbors import KernelDensity
import random
import pickle
import emd


class Marts():
    def __init__(self, 
                 params_MFEA = {'npop': 10,'ngen': 10,'mgen': 5,'psel': 0.5,'size_train': 100,'size_test': 50,}, 
                 feature_selection = True, 
                 distributive_version = False,
                 save_model = False,
                 decomposition = False,
                 max_lags = 5,
                 test_size = 100,
                 size_dataset_optimize_max_lags = 3
                 ):
        self.params_MFEA = params_MFEA
        self.feature_selection = feature_selection
        self.distributive_version = distributive_version
        self.save_model = save_model
        self.decomposition = decomposition
        self.G_list = {}
        self.dict_variables = {}
        self.dict_datasets_train = {}
        self.max_lags = max_lags
        self.target = ''
        self.imfs = []
        self.test = []
        self.test_size = test_size
        self.size_dataset_optimize_max_lags = size_dataset_optimize_max_lags


    def fit(self, dataset, target):
        
        retraining = {
            'tag': False}
        
        if self.distributive_version:
            num_cpu = os.cpu_count()
            
            if not ray.is_initialized():
                context = ray.init(num_cpus=num_cpu)
                print(context.dashboard_url)
                
        self.target = target
        
        #Empirical Mode Decomposition
        if self.decomposition:
            imf = emd.sift.sift(dataset[self.target].values)
            self.imfs = pd.DataFrame(imf, columns=(["IMF"+str(i) for i in range(1,imf.shape[1]+1)]))
            dataset = pd.concat([dataset,self.imfs], axis=1)
        
        train = dataset.loc[:dataset.shape[0]-self.test_size+1]
        self.test = dataset.loc[dataset.shape[0]-self.test_size:]
        
        # FEATURE SELECTION LAYER
        self.max_lags = fs.optimize_max_lags(train.loc[:train.shape[0]/self.size_dataset_optimize_max_lags], self.target)
        print(f"Number of lags: {self.max_lags}")
        
        if self.feature_selection:
            self.G_list = fs.causal_graph(train, self.target, self.max_lags)
            print("Causal graph of variables")
            print(self.G_list[self.target])
        else:
            self.G_list = fs.complete_graph(train, self.target, self.max_lags)        
                
        variable_delete = []
        for var in list(train.columns.values):
            if var not in self.G_list:
                variable_delete.append(var)
    
        train = train.drop(variable_delete, axis=1)
        
        if variable_delete:
            print(f"Variables {variable_delete} were deleted because they did not have predictive lags.")
            
        num_model = 1
    
        # MODEL SELECTION LAYER
        self.dict_datasets_train = util.get_datasets_all(train, self.G_list, self.max_lags, self.distributive_version)
    
        self.dict_variables = mg.initialize_model_layer(num_model, self.dict_datasets_train, self.target, train, self.params_MFEA, self.distributive_version, self.dict_variables, retraining)
        
        m=0
        for variable in self.dict_datasets_train:
        #for m in range(num_model):
            self.dict_variables[variable][m]["trained_model"], self.dict_variables[variable][m]["residuals"] = mg.evaluate_model(self.dict_variables[variable][m], self.dict_datasets_train[variable]['X_train'], self.dict_datasets_train[variable]['y_train'])

            
        if self.save_model:
            with open('model.pickle', 'wb') as f:
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
        

        #Empirical Mode Decomposition
        # l = list(self.dict_variables[self.target][0]['hiperparam']['model_size'].keys())

        # list_imfs_target = list(filter(lambda x:'IMF' in x, l))
        # imf = self.imfs.loc[:test.shape[0]-1,:][list_imfs_target]
        
        # test = pd.concat([test,imf], axis=1)
        
        variable_delete = []
        for var in list(test.columns.values):
            if var not in self.G_list:
                variable_delete.append(var)
    
        test = test.drop(variable_delete, axis=1)
    
        model = self.dict_variables[self.target][0]['trained_model']

            
        df_results = pd.DataFrame()
        df_distributions = pd.DataFrame()
    
        # if step_ahead == 1:
        #     dict_datasets_test = util.get_datasets(test, self.G_list, self.max_lags, self.target)
        #     predictions_tree = np.array([tree.predict(dict_datasets_test['X_train']) for tree in model.estimators_])
        #     kde_predictions_tree = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(predictions_tree.reshape((-1,1)))
        #     residual = random.choice(self.dict_variables[self.target][0]['residuals'])
        #     df_results[0] = (kde_predictions_tree.sample(100)).mean() + residual
        #     df_distributions[0] = kde_predictions_tree
        #     print((kde_predictions_tree.sample(100)).mean() + residual)

    
        test.index = range(0,test.shape[0])
    
        for row in range(test.shape[0] - self.max_lags):
            
            block_all = test.loc[row:row + self.max_lags]
            
            ### EXOGENOUS PREDICTION LAYER
            if step_ahead > 1:
                if block_all.shape[1] != 1:
                    block = block_all.drop([self.target], axis=1)
                else:
                    block = block_all
                block_forecast = pf.exogenous_forecast(step_ahead, block, self.max_lags, self.target, self.dict_variables, self.G_list)
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
                    predictions_tree = np.array([tree.predict(X) for tree in model.estimators_])
                    kde_predictions_tree = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(predictions_tree.reshape((-1,1)))
                    residual = random.choice(self.dict_variables[self.target][0]['residuals'])
                    forecast = (kde_predictions_tree.sample(100)).mean() + residual
                
                try:
                    block_nan[self.target].iloc[self.max_lags + f] = forecast
                    distributions.append(kde_predictions_tree)
                except:
                    df = pd.DataFrame(np.full([1, block_nan.shape[1],], np.nan), columns = block_nan.columns.values)
                    block_nan = pd.concat([block_nan, df])
                    block_nan[self.target].iloc[self.max_lags + f] = forecast
                    distributions.append(kde_predictions_tree)
                    
            block_nan.index = range(0,block_nan.shape[0])
            df_results[row] = block_nan[self.target].iloc[block_nan.shape[0]-step_ahead:].values
            df_distributions[row] = distributions
                
       
        if self.distributive_version:
            if ray.is_initialized():
                ray.shutdown() 
                
        return df_results, df_distributions
        

    

    def retraining(self, train):  

        retraining1 = {
            'tag': True,
            'hp': self.dict_variables}
                
        variable_delete = []
        for var in list(train.columns.values):
            if var not in self.G_list:
                variable_delete.append(var)
    
        train = train.drop(variable_delete, axis=1)
        
        num_model = 1
    
        # MODEL SELECTION LAYER
        self.dict_datasets_train = util.get_datasets_all(train, self.G_list, self.max_lags, self.distributive_version)
    
        self.dict_variables = mg.initialize_model_layer(num_model, self.dict_datasets_train, self.target, train, self.params_MFEA, self.distributive_version, retraining1)
        
        m=0
        for variable in self.dict_datasets_train:
            self.dict_variables[variable][m]["trained_model"], self.dict_variables[variable][m]["residuals"] = mg.evaluate_model(self.dict_variables[variable][m], self.dict_datasets_train[variable]['X_train'], self.dict_datasets_train[variable]['y_train'])
            
        if self.save_model:
            with open('model.pickle', 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

                
        