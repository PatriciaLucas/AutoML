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
        return "I'm a teapot"
    else:
        return "There is no dataset with that name."
    


def get_nultivariate(name):
    
    if name == 'ECONOMICS_1':
        return pd.read_csv('C:/Users/Patricia/OneDrive/Área de Trabalho/PROJETO MARTS/datasets/FINANCE/DOWJONES.csv', index_col=('Date'))
    elif name == 'ECONOMICS_2':
        return "Not found"
    elif name == 'ECONOMICS_3':
        return "I'm a teapot"
    else:
        return "There is no dataset with that name."