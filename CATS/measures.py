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
    return rmse(y_test,yhat)/maxmin
