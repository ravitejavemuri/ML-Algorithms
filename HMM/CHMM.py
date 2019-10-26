# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:31:39 2019

@author: Ravi
"""

import numpy as np
import pandas as pd
from GMM_Scr import GMM

        
def Alpha(D,pi,trp,emp):
    
    alpha = np.zeros((D.shape)) 
    alpha_ini = np.zeros((D.shape))
    # calculating alpha as per the formule
    for i in range(len(emp)):
        for j in range(len(pi)):
            alpha_ini[i,j] = pi[j] * emp[i]
    #print('alpha is', alpha_ini[91].dot(trp[:, 0])*emp[92])          
    for p in range(0,D.shape[0]):
        for q in range(trp.shape[0]):
            alpha[p,q] =alpha_ini[p - 1].dot(trp[q,:])*emp[p]
   
    return alpha


def Beta(D, trp, emp):
     # calculating alpha as per the formule
    print('In Beta')
    beta = np.zeros((D.shape[0],trp.shape[0] ))
    beta[D.shape[0] - 1] = np.ones((trp.shape[0]))
    for i in range(D.shape[0] - 2, -1, -1):
        for j in range(trp.shape[0]):
            beta[i,j] = (beta[i + 1]*emp[i+1]).dot(trp[:,j])   
    return beta

def Epsilon(D,trp,emp,alpha,beta,gamma):
    print('In Epsilon')
    M= trp.shape[0]
    T = len(D)
    for n in range(10):# iterations for convergance of epsilon
        epi = np.zeros((M,M,T-1))
        for i in range(T - 1):
            denominator = np.dot(np.dot(alpha[i, :].T, trp) * emp[i+1].T, beta[i + 1, :])
            for j in range(M):
                numerator = alpha[i, j] * trp[j, :] * emp[i+1].T * beta[i + 1, :].T
                epi[j, :, i] = numerator / denominator
                #print('epi', epi)
        gamma = np.sum(epi, axis=1)
        trp = np.sum(epi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
    return trp


def EM(alpha,beta,emp,GMM_Data,D):
    means = np.array(GMM_Data[0]) 
    co_var = np.array(GMM_Data[1]) 
    print('in gamma',D[2])
   
    numarator = alpha*beta
    #print('numarator', numarator)
    
    for i in range(0, alpha.shape[0]):
        denominator = np.sum(numarator[i, :])
        
    #print('denominator', denominator)
    c_jk = GMM_Data[2]
    for n in range(100):
        """E-Step"""
        gmm_numerator = c_jk * GMM_Data[5] # randomely initialized c_jk * normal_densities from GMM
        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
             gmm_part = gmm_numerator[j,:]/emp[i]
        gamma = (numarator/denominator * gmm_part)
        cjk_denm = np.sum(gamma, axis=1)
        cjk_denm = np.sum(cjk_denm, axis=0)
        cjk_num = np.sum(gamma, axis = 0)
        c_jk = cjk_num/cjk_denm
        
        """M-Step"""
        for i in range(alpha.shape[1]):
            unit_gamma = (gamma.T)[i]
            denm = np.dot(unit_gamma.T, np.ones(alpha.shape[0]))
            means = np.dot(unit_gamma.T, D)/denm
            
            difference = D - np.tile(means[i], (D.shape[0], 1))
            co_var =  np.dot(np.multiply(unit_gamma.reshape(D.shape[0],1), difference).T, difference) / denm
    return gamma, c_jk, means, co_var


data = pd.read_csv('iris.csv')
Obs1 = data['sepal.length'].values
Obs2 = data['sepal.width'].values

D = np.array(list(zip(Obs1,Obs2)))
trp =  np.array([(0.1, 0.9),(0.8,0.2)])
pi = np.array([0.3,0.7])

#print('D is ',np.ones(D.shape[0]).shape)
GMM = GMM(D,n_components = 2)
GMM_Data = GMM.fit(D)

#print('log_likelihood values', GMM_Data[3])
"""
Considering the log_vector from the GMM's EM procedure after convergence 
and initialising the emission probability  
What I learned : probability desities can be >> 1
"""
emp = GMM_Data[3]


#print('emp is', emp.shape, np.sum(emp[:,0]))
alpha = Alpha(D,pi,trp,emp)
#print('alpha is', alpha)
beta = Beta(D, trp, emp)
#print('beta is', beta)
gamma, c_jk, means, co_var  = EM(alpha, beta, emp, GMM_Data,D)
epsilon = Epsilon(D,trp,emp,alpha,beta,gamma)

state_data = np.zeros((D.shape)) 
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        state_data[i] = emp[i] * epsilon[j]

