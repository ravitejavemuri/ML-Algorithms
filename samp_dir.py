# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:15:28 2020

@author: GamaPSH
"""

# import pandas as pd
import numpy as np

y = np.array(([2, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12]))
alpha = np.array(([5, 7, 8, ]))
y_n = np.zeros((len(y), ))
x = np.zeros((len(y), ))
for i in range(len(y)): 
    y_n[i] = np.power(y[i], alpha[0]-1) * np.exp(-y[i]) / np.math.factorial(alpha[0]-1)

for j in range((len(y))):
    x[j] = y_n[j]/np.sum(y_n)
dirichlet = np.sum(x)
