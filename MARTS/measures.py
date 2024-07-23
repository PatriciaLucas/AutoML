# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:23:36 2023

@author: Patricia
"""

import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error



class Measures():
    def __init__(self, model):
        self.model = model


    def rmse(self, y_test,yhat):
        
        """
        Root Mean Squared Error 
        :return: 
        """
        return math.sqrt(mean_squared_error(y_test,yhat))
    
    def nrmse(self, real, forecast):
        
        """
        Root Mean Squared Error 
        :return: 
        """
        maxmin = abs(np.max(real) - np.min(forecast))
        
        if maxmin == 0:
            maxmin = 0.00001
        
        return self.rmse(real, forecast)/maxmin

    
    def mape(self, real, forecast):
        
        """
        Mean Average Percentual Error
        :return: 
        """ 

        return np.mean(np.abs(np.divide(np.subtract(real.values, forecast.values), forecast.values))) * 100
    
    
    def theil(self, real, forecast, step):
        """
        Theil's U Statistic

        :return: 
        """
        step = step + 1
        l = forecast.size
        l = 2 if l == 1 else l

        naive = []
        y = []
        for k in np.arange(0, l - step):
            y.append(np.subtract(forecast[k], real[k]) ** 2)
            naive.append(np.subtract(real[k + step], real[k]) ** 2)
        
        return np.sqrt(np.divide(np.nansum(y), np.nansum(naive)))
    
    
    def score(self, real, forecast):
        measures = {
            'rmse': [],
            'nrmse': [],
            'mape': [],
            'theil': []
            }
        

        for step in range(0,forecast.shape[0]):
            print(real[self.model.target][self.model.max_lags+step:].shape)
            measures['nrmse'].append(self.nrmse(real[self.model.target][self.model.max_lags+step:], forecast.loc[step,:forecast.shape[1]-(step+1)].to_frame()))
            measures['rmse'].append(self.rmse(real[self.model.target][self.model.max_lags+step:], forecast.loc[step,:forecast.shape[1]-(step+1)].to_frame()))
            measures['mape'].append(self.mape(real[self.model.target][self.model.max_lags+step:], forecast.loc[step,:forecast.shape[1]-(step+1)].to_frame()))
            measures['theil'].append(self.theil(real[self.model.target][self.model.max_lags+step:].values, forecast.loc[step,:forecast.shape[1]-(step+1)].values, step))

        return measures

