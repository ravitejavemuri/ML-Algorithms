# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:13:09 2019

Psudo : 
    1.Get a Dataset
    2.arbitarily choose K centroids in random
    3.Assign the closest data points by distance to a centroid/cluster
    4.Compute mean of the datapoints in the clusters excluding the centroids
    5.The mean would be the new centroid and repeat from step 3 until the centroid doesnt change.
@author: Ravi
"""

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the data sets
data =  pd.read_csv('data.csv')
#print(data.shape)
data.head()

#Plotting the values
d1= data['Dset1'].values
d2= data['Dset2'].values
X=np.array(list(zip(d1,d2)))
print('x iss',X)
plt.scatter(d1, d2, c='blue', s=7)


#Distance
def dist(a, b, ax=1):
    return np.linalg.norm(a-b, axis=ax)

#Picking centroids at random
k=4
C_x = np.random.randint(0, np.max(X)-20, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
C= np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)

#plotting with cetroids
plt.scatter(d1,d2, c ='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')


#Storing the value of centroids when it updates
C_old = np.zeros(C.shape)
#print(C_old)
clusters = np.zeros(len(X))
#distance between the new centroid and old centroid
Cdist = dist(C,C_old, None)

while Cdist != 0 :
    
    for i in range(len(X)):
        print( 'x i is',X[i])
        distances = dist(X[i], C)
        print(distances)
        cluster = np.argmin(distances)
        clusters[i] = cluster   
    #storing the old centroid
    C_old = deepcopy(C)
    #finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        #print(points)
        C[i] = np.mean(points, axis=0)
    Cdist = dist(C, C_old, None)
    
colors = ['r','g','b','y','c','m']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X))if clusters[j] == i])
    ax.scatter(points[:, 0], points[:,1], s=7, c=colors[i])
ax.scatter(C[:,0], C[:, 1], marker='*', s=200, c='#050505')
     
     
     

