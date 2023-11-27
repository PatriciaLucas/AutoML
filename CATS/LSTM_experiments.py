import sqlite3
import contextlib
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import random
import math
from operator import itemgetter
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from keras.layers import Dense, Flatten, SpatialDropout1D, Activation, Add, BatchNormalization, LSTM, Input, Dropout
from keras.constraints import unit_norm
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
import keras_tuner as kt
from keras.callbacks import EarlyStopping


def organize_dataset(dataset, G, max_lags, target):
    dataset.index = range(0,dataset.shape[0])
    lags = G.where(G).stack().index.tolist()
    y = dataset.loc[max_lags:]
    y.index = range(0,y.shape[0])
    try:
        for row in range(0,y.shape[0]):
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
    except:
        print("O PCMCI não encontrou links causais para a variável "+target)
        print("Aumente o número de lags observados.")

    return X, y

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

def get_datasets(dataset, G_list, max_lags, target):
    data = dict.fromkeys({"X": None, "y": None})
    data["X"], data["y"] = organize_dataset(dataset, G_list[target], max_lags, target)
    return data

def create_graph(dataset, target, max_lags):
    G_list = dict.fromkeys(list(dataset.columns.values), {})
    for var in G_list:
      G = pd.DataFrame(np.nan, index = np.arange(0,max_lags), columns = dataset.columns.values)
      for n in dataset.columns.values:
        G[n].loc[:] = True
      G.index = range(1,G.shape[0]+1)
      G_list[var] = G
    return G_list

def execute_insert(sql,data,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro data: dados que serão inseridos no banco de dados
    :parametro database_path: caminho para o banco de dados
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql,data)
                return cursor.fetchall()


def execute(sql,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro database_path: caminho para o banco de dados
    :return: dataframe com os valores retornados pela consulta sql
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql)
                return cursor.fetchall()

def rmse(y_test,yhat):

    """
    Root Mean Squared Error
    :return:
    """
    return math.sqrt(mean_squared_error(y_test,yhat))

def nrmse(y_test,yhat):

    """
    Root Mean Squared Error
    :return:
    """
    
    maxmin = np.max(y_test) - np.min(y_test)
    if maxmin != 0:
        return rmse(y_test,yhat)/maxmin
    else:
        return 0.0

def fit_deep(model_name, dataset):
  X_train = dataset['X'].values
  Y_train = dataset['y'].values

  X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
  X_train=np.asarray(X_train).astype(np.int)
  Y_train=np.asarray(Y_train).astype(np.int)

  stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

  if model_name == 'LSTM':

    def build_model(hp):
        model = Sequential()
        model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32),return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
        for i in range(hp.Int('n_layers', 1, 10)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
        model.add(LSTM(hp.Int('layer_2_neurons',min_value=32,max_value=512,step=32)))
        model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
        model.add(Dense(Y_train.shape[1], activation=hp.Choice('dense_activation',values=['relu', 'linear'],default='relu')))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
        return model

    '''
    tuner = RandomSearch(
        build_model,
        objective='mse',
        max_trials=30,
        executions_per_trial=1,
        overwrite=True
        )
    '''
    tuner = Hyperband(
        build_model,
        objective='mse',
        max_epochs=100,
        factor=5,
        hyperband_iterations=1,
        overwrite=True
    )


    tuner.search(
            x=X_train,
            y=Y_train,
            epochs=100,
            batch_size=100,
            verbose=0,
            callbacks=[stop_early]
    )


  return tuner


def predict(dataset, model, step_ahead, max_lags, G_list, target):
  df_results = pd.DataFrame(columns = list(range(0,step_ahead)))

  if step_ahead > 0:

      dataset.index = range(0,dataset.shape[0])

      for row in range(dataset.shape[0]-max_lags):
          block = dataset.loc[row:row+max_lags-1]

          # 1ª previsão
          X_input = organize_block(block, G_list[target], max_lags)
          X_input = np.reshape(X_input.values, (1,) + X_input.shape)
          X_input=np.asarray(X_input).astype(np.float)
          forecast = model.predict(X_input)[0]
          df_results.loc[row, 0] = forecast[dataset.columns.get_loc(target)]

          if step_ahead > 1:
            for step in range(1,step_ahead):

                block.loc[len(block)] = forecast
                block = block.drop([0])
                block.index = range(0,block.shape[0])

                X_input = organize_block(block, G_list[target], max_lags)
                X_input = np.reshape(X_input.values, (1,) + X_input.shape)
                X_input=np.asarray(X_input).astype(np.float)
                forecast = model.predict(X_input)[0]

                df_results.loc[row, step] = forecast[dataset.columns.get_loc(target)]

  return df_results

def execute_lstm(name_dataset, dataset, target, step_ahead, max_lags, database_path):

    execute("CREATE TABLE IF NOT EXISTS results(name_dataset TEXT, time FLOAT, max_lags INT, HPO BLOB, yhats BLOB, test BLOB, nrmse FLOAT)", database_path)

    start_time = time.time()
    #Normalização dos dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    dataset=pd.DataFrame(dataset, columns=data.columns.values) 

    #Organização dos dados de acordo com os lags
    G_list = create_graph(dataset, target, max_lags)

    train = get_datasets(dataset.loc[:2000], G_list, max_lags, target)

    #Treinamento e HPO
    tuner = fit_deep('LSTM', train)
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    model = tuner.hypermodel.build(best_hps)

    train = get_datasets(dataset.loc[:dataset.shape[0]-201], G_list, max_lags, target)

    X_train = train['X'].values
    Y_train = train['y'].values

    X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
    X_train=np.asarray(X_train).astype(np.int)
    Y_train=np.asarray(Y_train).astype(np.int)
    model.fit(X_train,Y_train, epochs = 100, verbose=1)

    hiperparams = model.summary()

    test = dataset.loc[dataset.shape[0]-200:]
    df_results = predict(test, model, step_ahead, max_lags, G_list, target)

    #Reverte a normalização
    test = scaler.inverse_transform(test)
    df_results = scaler.inverse_transform(df_results)

    #Teste
    runtime = round(time.time() - start_time, 2)

    erro = []

    if step_ahead == 1:
        erro.append(nrmse(test[target][max_lags:], df_results[0]))
    else:
        for i in range(df_results.shape[1]):
            erro.append(nrmse(test[target][max_lags+i:],df_results[i][:df_results.shape[0]-i]))


    #Salva no banco de dados
    execute_insert("INSERT INTO results VALUES(?, ?, ?, ?, ?, ?, ?)", (name_dataset, \
                                                                      runtime,
                                                                      max_lags,
                                                                      np.array(hiperparams).tostring(),
                                                                      df_results.to_numpy().tostring(),
                                                                      test[target].to_numpy().tostring(),
                                                                      np.array(erro).tostring()),
                  database_path)

    return


def run(datasets, target, num_experiments=10):
  step_ahead = 10
  max_lags = 10
  database_path = 'bd_lstm.db'

  for d in range(len(datasets)):
    print("Base de dados "+datasets[d])

    data = pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/CATS/datasets/'+datasets[d]+'.csv', on_bad_lines='skip')[:10000]

    for e in range(num_experiments):

      print("Experimento "+str(e))
      execute_lstm(datasets[d], data, target[d], step_ahead, max_lags, database_path)

  return
