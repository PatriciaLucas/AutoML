# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:23:44 2024

@author: Patricia
"""

import matplotlib.pyplot as plt


def plot_ahead(steps, real, forecasts):
    real.index = range(0,real.shape[0])
    fig, ax = plt.subplots(layout='constrained')
    
    real = real[10+0:]
    real.index = range(0,real.shape[0])
    plt.plot(real, color='black')
    
    for step in range(steps):
        plt.plot(forecasts.loc[step,:forecasts.shape[1]-(step+1)].to_frame())
        
        



