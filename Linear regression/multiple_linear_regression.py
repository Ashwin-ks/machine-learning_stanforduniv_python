# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:23:24 2016

@author: COMP
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path=os.getcwd()+'\ex1data2.txt'
data2=pd.read_csv(path,header=None,names=['Size','Bedrooms#','Price'])
#normalization
data2 = (data2 - data2.mean()) / data2.std()
# add ones column
def CostFunc(x,y,theta):
    inner= np.power(((x * theta.T) - y),2)
    cost=np.sum(inner)/(2*len(x))
    return cost
    
def GradDesc(x,y,theta,alpha,iters):    
    t=np.matrix(np.zeros(theta.shape))
    params=int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error=((x*theta.T)-y)
        for j in range(params):
            a=np.multiply(error,x[:,j])   ####np.multiply as we need elementwise multiplication not matrix multiply(a*b is matrix multiply for np.matrix)
            t[0,j]-=(alpha/len(x))*np.sum(a)
        theta=t
        cost[i]=CostFunc(x,y,theta)
    return theta,cost    
    
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))
alpha=0.01   ##select suitable alpha based on trial-error and plotting iterations vs cost fn
iters=1000       
colsize=data2.shape[1]
theta2=np.matrix(np.array([0,0,0]))     
print('Initial Cost :',CostFunc(X2,y2,theta2))
# perform linear regression on the data set
g2, cost2 = GradDesc(X2, y2, theta2, alpha, iters)

print ('Theta',g2)
print ('End cost value after linear regression',CostFunc(X2,y2,g2))

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')