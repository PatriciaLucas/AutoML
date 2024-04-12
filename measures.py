# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:23:36 2023

@author: Patricia
"""

import math
import numpy as np
from sklearn.metrics import mean_squared_error

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
    
    if maxmin == 0:
        maxmin = 0.00001
    
    return rmse(y_test,yhat)/maxmin

