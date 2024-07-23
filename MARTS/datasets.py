# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:46:19 2024

@author: Patricia
"""

import pandas as pd


def get_univariate(name, target):
    
    if name == 'ECONOMICS_1':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_1.csv', index_col=('Date'))[target]
    elif name == 'ECONOMICS_2':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_2.csv', index_col=('Date'))[target]
    elif name == 'ECONOMICS_3':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_3.csv', index_col=('Date'))[target]
    elif name == 'ECONOMICS_4':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_4.csv', index_col=('Date'))[target]
    elif name == 'ECONOMICS_5':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_5.csv', index_col=('Date'))[target]
    elif name == 'ECONOMICS_6':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_6.csv', index_col=('Date'))[target]
    elif name == 'ECONOMICS_7':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_7.csv', index_col=('Date'))[target]
    
    
    
    
    
    
    
    else:
        return "There is no dataset with that name."
    


def get_multivariate(name):
    
    if name == 'ECONOMICS_1':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_1.csv', index_col=('Date'))
    elif name == 'ECONOMICS_2':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_2.csv', index_col=('Date'))
    elif name == 'ECONOMICS_3':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_3.csv', index_col=('Date'))
    elif name == 'ECONOMICS_4':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_4.csv', index_col=('Date'))
    elif name == 'ECONOMICS_5':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_5.csv', index_col=('Date'))
    elif name == 'ECONOMICS_6':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_6.csv', index_col=('Date'))
    elif name == 'ECONOMICS_7':
        return pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/AutoML/main/Datasets/ECONOMICS_7.csv', index_col=('Date'))
    else:
        return "There is no dataset with that name."