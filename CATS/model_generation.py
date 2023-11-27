import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import sys
sys.path.append('./')
import CATS.MFEA


def random_model():
    name_model = np.random.choice(['RandomForest'], size=1, replace=False)[0]
    return name_model


def initialize_model_layer(num_model, dict_datasets_train, target, series):
    # Dictionary that stores the ensembles of each variable in the database.
    dict_variables = dict.fromkeys(list(dict_datasets_train.keys()), {})
    
    hp = MFEA.GeneticAlgorithm(dict_datasets_train, series)

    for variable in dict_variables:
        dict_ensemble = dict.fromkeys(list(range(0, num_model)), None)
        for m in range(num_model):
            #Dictionary that stores the information of each model.
            dict_model =  {"name": random_model(), "hiperparam": hp, "trained_model": None, "residuals":None}
            #Dictionary that stores the models of each ensemble.
            dict_ensemble[m] = dict_model
        dict_variables[variable] = dict_ensemble
            
    
    return dict_variables
    


def evaluate_model(dict_model, X_train, y_train):
    if dict_model['name'] == 'RandomForest':
        model = RandomForestRegressor(n_estimators = dict_model['hiperparam']['n_estimators'], max_features = dict_model['hiperparam']['max_features'],
                                      min_samples_leaf = dict_model['hiperparam']['min_samples_leaf'], n_jobs = -1)
    else:
        model = LinearRegression()
        
    model.fit(X_train, y_train)
    forecasts = model.predict(X_train)
    max_lags = y_train.shape[0] - forecasts.shape[0]
    residuals = y_train[max_lags:].values - forecasts
    return model, residuals
