# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:01:35 2024

@author: Patricia
"""

import save_database as sd
import marts
import measures
import util
import pandas as pd
import datasets
import MFEA



data = datasets.get_multivariate('ECONOMICS_1').iloc[:2000]
data.index = range(0,data.shape[0])

#data = data.loc[:1000]

target = 'AVG'
step_ahead = 3
windows_size = .5
test_size = 0
w = int(data.shape[0] * windows_size)
d = int(.2 * w)
i=0

dataset = data[i*d:(i*d)+w]

train, test = dataset.head(int(dataset.shape[0]-test_size)), dataset.tail(test_size)

j_MFEA = (train.shape[0])/3

params_MFEA = {
    'npop': 4,
    'ngen': 2,
    'mgen': 5,
    'psel': 0.5,
    'size_train': j_MFEA - (j_MFEA*0.2),
    'size_test': j_MFEA*0.2
    }

model = marts.Marts(params_MFEA = params_MFEA, feature_selection = True, distributive_version = False, 
                    save_model = False, decomposition = False, test_size=test_size, size_dataset_optimize_max_lags=3,
                    optimize_hiperparams = True)
model.fit(train, target)


df_results = model.predict_ahead_mult(step_ahead=1, target=target)

a = model.dict_datasets_test


test = model.target_test
import matplotlib.pyplot as plt
plt.plot(range(0,test.shape[0]),test)
plt.plot(range(0,test.shape[0]),df_results.values)


mea = measures.Measures(model)
results = mea.score(test, df_results.T)




