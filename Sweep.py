# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:19:44 2020

@author: GamaPSH
"""
import numpy as np
from sampling_parameters import Draw
from moves import Moves

class Sweep:
    def init(self,n_iter,n_components):
        self.n_iter = n_iter
        self.n_components = n_components
        self.drawObj = Draw()
        self.moveObj = Moves()
        self.moveObj.init(n_components)
        self.jp =None
        self.jp_old =None
        self.parameters = self.drawObj.sample_params(n_components)
    def main(self):
        print('main code goes here') 
        jp =  self.joint_proba()
        self.accept_reject()
        self.moves()
        return jp
    def joint_proba(self):
        alpha,K, A,Z,sigma,y = self.parameters 
        jp = np.zeros(Z.shape)
        print('self', self.n_components)
        for t in range(self.n_components):
            jp[t,:]=alpha[t]*K[t]*A[t,t]*Z[t,:]*sigma[t]*y[t] # confirm how to choose Z values
        print('sum', jp,np.sum(jp, axis=1))
        return jp
    def moves(self):
        print('moves here')
        move = self.moveObj
        move.update_A(1,2)
        move.update_sigma()
        #move.update_Z()
        #move.update_alpha()
        #move.split_comb()
        #move.birth_death()
        
        
    def accept_reject(self):
        print('accept reject here')
        
sweepObj = Sweep()
sweepObj.init(1,9)
jp = sweepObj.main()