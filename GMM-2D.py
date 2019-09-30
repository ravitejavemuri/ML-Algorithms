# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 23:19:49 2019

@author: Ravi
GMM with multi-dimesional data
"""
 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('dark_background')
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from scipy.stats import multivariate_normal

k=3; # number of gaussians
log_likelihoods = []
iterations=100;
#Creating a 2 Dimensional data set with 2 features with the help of sklearn datasets
features,samples = make_blobs(cluster_std=1.5,random_state=20,n_samples=200,centers=3)
#print('data is',features,samples)
features = np.dot(features,np.random.RandomState(0).randn(2,2))

#initialize a identity covar matrix 2x2
cov_id = np.identity(len(features[0]))
#creating a coordinate matrix from features to map
x,y = np.meshgrid(np.sort(features[:,0]),np.sort(features[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T # transpose of the data matrix

#initialize the mean, covar, pi
mean = np.random.randint(-5,10,size=(k,len(features[0])))
co_var = np.zeros((k,len(features[0]),len(features[0])))
for dim in range(len(co_var)):
            np.fill_diagonal(co_var[dim],4)
pi= np.ones(k)/k

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(features[:,0],features[:,1])
ax0.set_title('Initial state')
for m,c in zip(mean,co_var):
    c += cov_id
    multi_gauss = multivariate_normal(mean=m,cov=c)
    ax0.contour(np.sort(features[:,0]),np.sort(features[:,1]),multi_gauss.pdf(XY).reshape(len(features),len(features)),colors='white',alpha=0.5)
    ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
#print('covariance', c)

for i in range(iterations):
    """E Step"""
    resp = np.zeros((len(features),len(co_var))) # initializing the responsibility matrix nxk
    for m,co,p,r in zip(mean,co_var,pi,range(len(resp[0]))):
        #calculating the responsibility of each point 
        co+=cov_id
        mn = multivariate_normal(mean=m,cov=co)
        resp[:,r] = p*mn.pdf(features)/np.sum([pi_c*multivariate_normal(mean=mu_c,cov=cov_c).pdf(features) for pi_c,mu_c,cov_c in zip(pi,mean,co_var+cov_id)],axis=0)
    #print('calculated covar', co)
    
    """M Step"""
    mean = []
    co_var = []
    pi = []
    log_likelihood = []
    for c in range(len(resp[0])):
        #updating mean, co_var and pi values 
        m_c = np.sum(resp[:,c],axis=0)
        #print('Cov is ', co_var,m_c)  
        mu_c = (1/m_c)*np.sum(features*resp[:,c].reshape(len(features),1),axis=0)
        mean.append(mu_c)
        # Calculate the covariance matrix per source based on the new mean
        co_var.append(((1/m_c)*np.dot((np.array(resp[:,c]).reshape(len(features),1)*(features-mu_c)).T,(features-mu_c)))+cov_id)
        
       # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source 
        pi.append(m_c/np.sum(resp)) 
        """Log likelihood"""
    log_likelihoods.append(np.log(np.sum([k*multivariate_normal(mean[i],co_var[j]).pdf(features) for k,i,j in zip(pi,range(len(mean)),range(len(co_var)))])))

fig2 = plt.figure(figsize=(10,10))
ax1 = fig2.add_subplot(111) 
ax1.set_title('Log-Likelihood')
ax1.plot(range(0,iterations,1),log_likelihoods)

clust_test = [[5,0.5]] # Random sample
fig3 = plt.figure(figsize=(10,10))
ax2 = fig3.add_subplot(111)
ax2.scatter(features[:,0],features[:,1])
for m,c in zip(mean,co_var):
    multi_normal = multivariate_normal(mean=m,cov=c)
    ax2.contour(np.sort(features[:,0]),np.sort(features[:,1]),multi_normal.pdf(XY).reshape(len(features),len(features)),colors='yellow',alpha=0.5)
    ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    ax2.set_title('Final state')
    for y in clust_test:
        #print(y[0],y[1])
        ax2.scatter(y[0],y[1],c='black',zorder=10,s=100)
#Testing the cluster for random sample
prediction = []        
for m,c in zip(mean,co_var):  
    #print(c)
    prediction.append(multivariate_normal(mean=m,cov=c).pdf(clust_test)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(clust_test) for mean,cov in zip(mean,co_var)]))
