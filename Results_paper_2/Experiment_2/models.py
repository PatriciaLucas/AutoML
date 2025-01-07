# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:32:46 2024

@author: Patricia
"""
import pandas as pd
from AUTOTSF import feature_selection as fs
from AUTOTSF import model_generation as mg
from AUTOTSF import util
from AUTOTSF import forecast as fo
import numpy as np

def organize_dataset(dataset, G, max_lags, target):
    dataset.index = range(0,dataset.shape[0])
    lags =  G.where(G).stack().index.tolist()
    y = dataset.loc[max_lags:]
    y.index = range(0,y.shape[0])

    y_shape = y.shape[0]
    values = []
    for row in range(0,y_shape):
            bloco = dataset.iloc[row:max_lags+row]
            bloco.index = reversed(range(1,bloco.shape[0]+1))
            values.append([bloco.loc[lag[0], lag[1]] for lag in lags])

    X = pd.DataFrame(values, columns =[l[1] for l in lags])

    return X, y

def complete_graph(dataset, target, max_lags):

    G_list = dict.fromkeys(list(dataset.columns.values), {})

    for var in G_list:
        G = pd.DataFrame(True, index = np.arange(0,max_lags+1), columns = dataset.columns.values)
        if var != target: del G[target]
        G_list[var] = G.iloc[1:]

    return G_list

def get_datasets(dataset, G_list, max_lags, target):
    data = dict.fromkeys({"X_train": None, "y_train": None})
    data["X_train"], data["y_train"] = organize_dataset(dataset, G_list[target], max_lags, target)
    return data


def organize_block(dataset, G, max_lags):
    dataset.index = range(0,dataset.shape[0])
    lags = G.where(G).stack().index.tolist()

    for row in range(0,1):
        cols = []
        values = []
        bloco = dataset.iloc[row:max_lags+row]
        bloco.index = reversed(range(1,bloco.shape[0]+1))
        for lag in lags:
            if row == 0:
                cols.append(lag[1])
                X = pd.DataFrame(columns=cols)
            values.append(bloco[lag[1]].loc[lag[0]])

        X.loc[row,:] = values
    return X


def predict(dataset, model, step_ahead, max_lags, G_list, target):

  df_results = pd.DataFrame(columns = list(range(0,step_ahead)))
  block = dataset.loc[dataset.shape[0]-max_lags:]
  block.index = range(0,block.shape[0])

  X_input = organize_block(block, G_list[target], max_lags)
  #X_input = np.reshape(X_input.values, (1,) + X_input.shape)
  #X_input=np.asarray(X_input).astype(float)
  forecast = model.predict(X_input.values).reshape(dataset.shape[1]).tolist()
  df_results.loc[0, 0] = forecast[dataset.columns.get_loc(target)]

  for step in range(1,step_ahead):

    block.loc[len(block)] = forecast
    block = block.drop([0])
    block.index = range(0,block.shape[0])

    X_input = organize_block(block, G_list[target], max_lags)
    #X_input = np.reshape(X_input.values, (1,) + X_input.shape)
    #X_input=np.asarray(X_input).astype(float)
    forecast = model.predict(X_input.values).reshape(dataset.shape[1]).tolist()
    df_results.loc[0, step] = forecast[dataset.columns.get_loc(target)]

  return df_results        
        
        
def proposto(dataset, test_size, target):
    from AUTOTSF import autotsf
    dataset.index = range(0,dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size)

    j_MFEA = (train.shape[0])/3 
    
    j_MFEA = 1000 if j_MFEA > 1000 else j_MFEA
    params_MFEA = {
        'npop': 10,
        'ngen': 5,
        'mgen': 5,
        'psel': 0.5,
        'size_train': j_MFEA - (j_MFEA*0.2),
        'size_test': j_MFEA*0.2
        }
    
    model = autotsf.AUTOTSF(params_MFEA = params_MFEA, feature_selection = True, distributive_version = True, max_lags=20,
                        save_model = True, decomposition = False, test_size=0, size_dataset_optimize_max_lags=3,
                        optimize_hiperparams = True)
    model.fit(train, target)
    forecast = model.predict_ahead_mult(step_ahead=test.shape[0], target=target).values.reshape(-1)
    real = test[target].values.tolist()

    del model
    del dataset
    return real, forecast


def autogluon(dataset, test_size, target):
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    
    dataset['Time'] = pd.to_datetime(dataset.index)
    dataset.index = range(0, dataset.shape[0])
    dataset['id'] = ['1'] * len(dataset)
    data = TimeSeriesDataFrame.from_data_frame(dataset, id_column="id", timestamp_column="Time")
    train, test = data.head(int(data.shape[0]-test_size)), data.tail(test_size)
    model = TimeSeriesPredictor(target=target, 
                                prediction_length=test.shape[0], 
                                eval_metric="MASE",
                                verbosity=1,
                                log_to_file=False,
                                #freq = 'H'
                                )
    model.fit(train, presets=['best_quality'], time_limit=float(2*3600))
    forecast = model.predict(train)['mean'].to_list()
    real = test[target].values
    del model
    
    return real, forecast



def fedot(dataset, test_size, target):
    from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams 
    from fedot.core.data.data import InputData 
    from fedot.core.data.data_split import train_test_data_setup
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot import Fedot
    import numpy as np
    
    
    dataset.index = range(0,dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size)
    forecast_length = test_size
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length))
    
    input_data = InputData(idx=np.arange(0, len(train)),
                           features=train.values,
                           target=train.values,
                           task=task,
                           data_type=DataTypesEnum.ts)
    train_input, predict_input = train_test_data_setup(input_data)
    model = Fedot(problem='ts_forecasting', preset='ts', task_params=task.task_params,
                  num_of_generations=10, pop_size=20,  
                  n_jobs=-1, metric='mae', with_tuning=True,
                  )
    model.fit(features=train_input)
               
    forecast = model.predict(features=predict_input, in_sample=False)

    real = test[target].values
    del model
    return real, forecast


def pycaret(dataset, test_size, target):
    from pycaret.time_series import TSForecastingExperiment

    #dataset['index'] = pd.to_datetime(dataset.index)
    dataset.index = range(0, dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size)
    print(train.shape)
    exp_auto = TSForecastingExperiment()
    exp_auto.setup(
        data=train, target=target,
        enforce_exogenous=True,
        numeric_imputation_target="ffill", numeric_imputation_exogenous="ffill",
        session_id=42, 
    )

    real = test[target]
    
    best = exp_auto.compare_models(verbose=True)
    print(best)
    best_model = exp_auto.tune_model(
                        best,
                        choose_better=True,
                        n_iter=50,
                        fold=3,
                        search_algorithm="random",
                        tuner_verbose=True,
                    )
    
    forecast = exp_auto.predict_model(best_model, fh=30)

    return real.values, forecast.values.flatten()

def RF(dataset, test_size, target):
    from sklearn.ensemble import RandomForestRegressor
    dataset.index = range(0,dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size+20)
    G = complete_graph(train, target, 20)
    train = get_datasets(train, G, 20, target)
    model = RandomForestRegressor(bootstrap=True, n_jobs = -1)
    model.fit(train['X_train'].values, train['y_train'].values)
    forecast = predict(test, model, 30, 20, G, target)
    return test[target].values, forecast.values[0].tolist()

def XGBOOST(dataset, test_size, target):
    from xgboost import XGBRegressor
    dataset.index = range(0,dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size+20)
    G = complete_graph(train, target, 20)
    train = get_datasets(train, G, 20, target)
    model = XGBRegressor(bootstrap=True, n_jobs = -1)
    model.fit(train['X_train'].values, train['y_train'].values)
    forecast = predict(test, model, 30, 20, G, target)
    return test[target].values, forecast.values[0].tolist()


def LIGHTGBM(dataset, test_size, target):
    from sklearn.multioutput import MultiOutputRegressor
    from lightgbm import LGBMRegressor
    dataset.index = range(0,dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size+20)
    G = complete_graph(train, target, 20)
    train = get_datasets(train, G, 20, target)
    lgb_model = LGBMRegressor(n_jobs = -1)
    model = MultiOutputRegressor(lgb_model)
    model.fit(train['X_train'].values, train['y_train'].values)
    forecast = predict(test, model, 30, 20, G, target)
    return test[target].values, forecast.values[0].tolist()



def autots(dataset, test_size, target):
    from autots import AutoTS
    
    dataset['Time'] = pd.to_datetime(dataset.index)
    dataset.index = range(0, dataset.shape[0])
    train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size)   
    
    
    model = AutoTS(forecast_length=int(test.shape[0]), generation_timeout=60,
                   num_validations=0, max_generations=10,
                   model_list = 'multivariate',
                   n_jobs='auto',
                   verbose=-10)
    model.model_list
    model = model.fit(train, date_col='Time', value_col=target, id_col=None)
    best_model = model.best_model
    print(f"Best Model: {best_model['Model']}")
    print(f"Model Parameters: {best_model['ModelParameters']}")
    print(f"Transformation Parameters: {best_model['TransformationParameters']}")
    prediction = model.predict()
    forecast = prediction.forecast[target].to_list()
    real = test[target].values 
    del model
    return real, forecast





    