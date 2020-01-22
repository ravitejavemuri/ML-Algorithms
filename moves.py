# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 08:54:00 2020

@author: GamaPSH
"""

import numpy as np


class Moves: 
    def init(self, n_components):
        
        self.n_components = n_components
        self.n_jumps = np.zeros((n_components,n_components))
        self.sigma=None
        self.Z=None
        self.alpha=None
    def update_A(self,i,j):
        temp_n_jumps = self.n_jumps
        temp_n_jumps[i,j] =+1
        return temp_n_jumps
    def update_sigma(self):
        print('sigma here')
        temp_sigma = None        
    def update_Z(self,Obs_y,pi,):
        print('z here')
        temp_Z = np.zeros(Obs_y.shape)
        temp_Z[0,:] = pi[0]
        temp_Z[n_components-1,:] = 1
        for t in range(1,len(Obs_y)-1):
            print(t)
            print( t-1,t,t,t+1)
           
            Z[t,:] = A[t-1,t]*Obs_y[t]*A[t,t+1]       
        return Z
    def update_alpha(self):
        print('alpha here')
        
        temp_alpha = None
    def split_comb(self):
        print('split or combine')
    def birth_death(self):
        print('birth or death')
    