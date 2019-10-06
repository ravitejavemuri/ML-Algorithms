# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:53:56 2019
HMM Alorithm : Assuming 2 Hidden states and 3 observations
    1.Initilize the start(pi), transition(trp), emission probabilities(emp)
    2.Choose the observation data set(D)
    3.Calculate the Alpha( forward part ) using pi,D, trp, emp
    4.Calculate the Beta( backward part ) using D, trp,emp
    E:Step
    5.Calculate epsilon to maximize the probability epi(i,j) =alpha(i) at t * trp for ij * emp for j at t+1 * beta(j) at t+1/ SUM(alpha(i) at t * trp for ij * emp for j at t+1 * beta(j) at t+1)
    6.Calculate optimality criterion gamma = SUM(epi)
    M:Step
    7.Update the values of  trp, emp.
    

@author: Ravi
"""

import numpy as np
import pandas as pd

def Alpha(D,pi,trp,emp):
    print('In alpha',emp[:, D[0]])
    
    alpha = np.zeros((D.shape[0], 2)) # array of observations from two states
    
    # calculating alpha
    alpha[0, :] = pi * emp[:, D[0]]
    print('alpha is ', alpha[0])
    
    for i in range(1,D.shape[0]):
        for j in range(trp.shape[0]):
            alpha[i,j] = alpha[i - 1].dot(trp[:,j])*emp[j,D[i]]
    
    return alpha

def Beta(D, trp, emp):
    print('In Beta')
    beta = np.zeros((D.shape[0],trp.shape[0] ))
    beta[D.shape[0] - 1] = np.ones((trp.shape[0]))
    #print('beta is', beta)
    
    for i in range(D.shape[0] - 2, -1, -1):
        for j in range(trp.shape[0]):
            beta[i,j] = (beta[i + 1]*emp[:,D[i + 1]]).dot(trp[j,:])
            
    return beta
            
    


data = pd.read_csv('data_hmm.csv')
D = data['Observations'].values

#initial probability pi
pi = np.array([0.3,0.7])

#emission probability emp - 2 states 3 observations so 6 probabilities
emp = np.array([[0.345,0.987,0.112],[0.888, 0.456, 0.256]])

# Transition probabilities trp - 2 states so 4 probabilities
trp =  np.array([[0.9, 0.1],[0.6,0.4]])

alpha = Alpha(D,pi,trp,emp)
#print('alpha is', alpha)
beta = Beta(D, trp, emp)
#print('beta is', beta)

#Baum_welch algorithm
M= trp.shape[0]
T = len(D)


for n in range(100):
    epi = np.zeros((M,M,T-1))
    for i in range(T - 1):
        denominator = np.dot(np.dot(alpha[i, :].T, trp) * emp[:, D[i + 1]].T, beta[i + 1, :])
        for j in range(M):
            numerator = alpha[i, j] * trp[j, :] * emp[:, D[i + 1]].T * beta[i + 1, :].T
            epi[j, :, i] = numerator / denominator
    gamma = np.sum(epi, axis=1)
    trp = np.sum(epi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
    
    gamma = np.hstack((gamma, np.sum(epi[:, :, T - 2], axis=0).reshape((-1, 1))))
    
    K = emp.shape[1]
    denominator = np.sum(gamma, axis=1)
    for l in range(K):
        emp[:, l] = np.sum(gamma[:, D == l], axis=1)
    
    emp = np.divide(emp, denominator.reshape((-1, 1)))

print('trp is', trp )
print('emp is', emp)