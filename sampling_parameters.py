# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 14:03:28 2020

@author: Ravi
"""

import pandas as pd
import numpy as np

class Draw :
    
    def init(self, n_components):
        self.n_components = n_components
    def sample_K(self):        
        K = [1/i- 0 for i in np.arange(1,self.n_components+1)]
        return K
    def sample_A(self):
        """ using gamma dist to sample from dirichlet"""
        shape = self.n_components
        A = np.zeros((shape,shape))
        y_n = np.zeros((shape, ))
        alpha_dir = np.array(([5]))
        for n in range(shape): 
            row = np.random.rand(shape,1)# generating randoms to sample from gamma
            for i in range (shape):
                #gamma dist
                y_n[i] = np.power(row[i],alpha_dir[0]-1) * np.exp(-row[i]) / np.math.factorial(alpha_dir[0]-1)
                for j in range((len(row))) :
                    A[n,j] = y_n[j]/np.sum(y_n) # dirichlet values each row in A
                A_sum = np.sum(A, axis = 1) # check for row sum 1
        return A
    def sample_sigma(self, D):
        e_x = D
        sigma_sd = []
        exp_alpha = []
        exp_mean = 30*np.amax(e_x, axis=0)
        for i in range(len(e_x)):
            #print('exp', -1/exp_mean*e_x[i])
            exp_alpha.append( 1/exp_mean * np.exp(-1/exp_mean*e_x[i])) # pdf of negative exponential distribution
        sigma_sd.append( [1/exp_alpha[i]-0 for i in range(len(exp_alpha))])
        sigma_sd = np.asarray((sigma_sd))
        sigma_sd = np.sort(sigma_sd[0], axis=0) # sorting sigma in accending order
        return sigma_sd,np.array((exp_alpha))
    
    def multi_gauss_pdf(self,x,co_var,D):
         #print(x)
         centered = x - 0
         cov_inverse = np.linalg.inv(co_var)
         cov_det = np.linalg.det(co_var)
         exponent = np.dot(np.dot(centered.T, cov_inverse), centered)
         result = np.exp(-0.5 * exponent) / np.sqrt(np.absolute(cov_det * np.power(2 * np.pi, D.shape[1])))
         #print('res', result)
         return result
    
    def sample_normal(self, D):
        normal_densities = np.empty((D.shape[0], D.shape[1]), np.float)
        split = int(np.floor(D.shape[0]/D.shape[1]))
        D_split = [D[i:i+split] for i in range(0,D.shape[0], split)]
        co_var = []
        for i in range(D.shape[1]):
            co_var.append(np.cov(D_split[i].T))
            #print('co_var', co_var[0])
        co_var = np.array(co_var)
        for p in range(len(D)):
            x=D[p]
            for j in range(D.shape[1]):
                normal_densities[p][j] = self.multi_gauss_pdf(x,co_var[j],D)
        return normal_densities
    def sample_allocations(self,n_components,K,A,sigma,Obs_y,pi):
        Z = np.zeros(Obs_y.shape)
        Z[0,:] = pi[0]
        Z[n_components-1,:] = 1
        #print("shape is Z", Z.shape)
        """
        #Z = np.array((sigma.shape))
        print('sample allocations Z')
        for n in range(len(Obs_y)):
                for j in range(len(K)):
                    #print(A[i,j])
                    Z_1 = A[n,:]/sigma[n]
                    Z_2 = np.power(Obs_y[n], 2) / 2*np.power(sigma[n],2)
                    Z_exp = np.exp(- Z_2)
                    Z.append( Z_1 * Z_exp)
                    print(Z)
        """
        for t in range(1,len(Obs_y)-1):
            print(t)
            print( t-1,t,t,t+1)
           
            Z[t,:] = A[t-1,t]*Obs_y[t]*A[t,t+1] # not Obs_y it's posterior of Obs_y 
        return Z
                    
        
        
    def sample_params(self,n_components):
        D = pd.read_csv('data.csv')
        D = np.array((D))
        drawObj = Draw()
        drawObj.init(n_components)
        K = drawObj.sample_K()
        A = drawObj.sample_A()
        pi =  A[0,:]
        sigma,alpha = drawObj.sample_sigma(D)
        Obs_y = drawObj.sample_normal(D)
        Z = self.sample_allocations(n_components,K,A,sigma,Obs_y,pi)
        return alpha,K,A,Z,sigma,Obs_y
        
   
res = Draw()
ress = res.sample_params(9)




"""
#defining pi - stationary vector

#sampling componets K from uniform distribution
a = 0
K = [1/i-a for i in np.arange(1,n_components)]

# sampling A from dirichlet distribution

A = np.zeros((n_components,n_components))
y_n = np.zeros((n_components, ))
alpha_dir = np.array(([5]))
for n in range(n_components): 
    row = np.random.rand(n_components,1)# generating randoms to sample from gamma
    for i in range (n_components):
        #gamma dist
        y_n[i] = np.power(row[i],alpha_dir[0]-1) * np.exp(-row[i]) / np.math.factorial(alpha_dir[0]-1)
    for j in range((len(row))) :
        A[n,j] = y_n[j]/np.sum(y_n) # dirichlet values each row in A
    A_sum = np.sum(A, axis =1)# check for row sum 1
    

# sampling alpha from negative exponential distribution

e_x = data
sigma_sd = []
exp_alpha = []
for i in range(len(e_x)):
    exp_mean = 20*np.amax(e_x[i])
    #print('exp', -1/exp_mean*e_x[i])
    exp_alpha.append( 1/exp_mean * np.exp(-1/exp_mean*e_x[i])) # pdf of negative exponential distribution
sigma_sd.append( [1/exp_alpha[i]-a for i in range(len(exp_alpha))])
sigma_sd = np.asarray(sigma_sd)
sigma_sd = np.sort(sigma_sd[0], axis=0) # sorting sigma in accending order


#Sweeps code

n_jumps = np.zeros((A.shape))

for i in range(len(A)):
    for j in range(len(A)):
        if(i!=j):
            n_jumps[i,j] =+1
"""
