import CATS.feature_selection as fs
import CATS.model_generation as mg
import CATS.organization as org
import CATS.montecarlo_forecast as bf
import pandas as pd
import numpy as np


def fit(train, target):
    num_model = 1
    
    
    # FEATURE SELECTION LAYER
    max_lags = fs.optimize_max_lags(train, target)
    
    G_list = fs.causal_graph(train, [target], max_lags)
    
    
    variable_delete = []
    for var in list(train.columns.values):
        if var not in G_list:
            variable_delete.append(var)

    train = train.drop(variable_delete, axis=1)

    # MODEL SELECTION LAYER
    dict_datasets_train = org.get_datasets_all(train, G_list, max_lags)

    dict_variables = mg.initialize_model_layer(num_model, dict_datasets_train, target, train)

    for variable in dict_datasets_train:
        for m in range(num_model):
            dict_variables[variable][m]["trained_model"], dict_variables[variable][m]["residuals"] = mg.evaluate_model(dict_variables[variable][m], dict_datasets_train[variable]['X'], dict_datasets_train[variable]['y'])

    return G_list, dict_variables, dict_datasets_train, max_lags




# ENDOGENOUS PREDICTION LAYER
def predict(test, step_ahead, bootstrap_size, dict_variables, G_list, max_lags, target):
    
    variable_delete = []
    for var in list(test.columns.values):
        if var not in G_list:
            variable_delete.append(var)

    test = test.drop(variable_delete, axis=1)

    model = dict_variables[target][0]['trained_model']
    
    df_results = pd.DataFrame()

    if step_ahead == 1:
        dict_datasets_test = org.get_datasets(test, G_list, max_lags, target)
        forecast = model.predict(dict_datasets_test['X'])
        df_results[0] = forecast
    else:

        test.index = range(0,test.shape[0])
    
        for row in range(test.shape[0]-max_lags):
            block_all = test.loc[row:row+max_lags]
            if block_all.shape[1] != 1:
                block = block_all.drop([target], axis=1)
            else:
                block = block_all
                
            # EXOGENOUS PREDICTION LAYER
            block_forecast = bf.montecarlo_forecast(step_ahead, block, max_lags, bootstrap_size, target, dict_variables, G_list)
            
            block_nan = pd.concat([block_all, block_forecast])
            block_nan.index = range(0,block_nan.shape[0])
            block_nan = block_nan.drop([0])
            block_nan.index = range(0,block_nan.shape[0])
            for f in range(0,step_ahead):
                X = org.organize_block(block_nan.iloc[f:], G_list[target], max_lags)
                forecast = model.predict(X)[0]
                
                try:
                    block_nan[target].iloc[max_lags+f] = forecast
                except:
                    df = pd.DataFrame(np.full([1, block_nan.shape[1],], np.nan), columns = block_nan.columns.values)
                    block_nan = pd.concat([block_nan, df])
                    block_nan[target].iloc[max_lags+f] = forecast
                    
            block_nan.index = range(0,block_nan.shape[0])
            df_results[row] = block_nan[target].iloc[block_nan.shape[0]-step_ahead:].values
        
        
    return df_results

