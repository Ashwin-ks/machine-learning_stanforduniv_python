
#implement simple linear regression using gradient descent
#and apply it to an example problem without scikit-learn

#implementing linear regression with one variable to predict profits
#for a food truck. Suppose you are the CEO of a restaurant franchise 
#and are considering different cities for opening a new outlet. 
#The chain already has trucks in various cities and you have data 
#for profits and populations from the cities.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

path=os.getcwd()+'\ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
#data.head()
#data.describe()
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))

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
        
alpha=0.01   ##select suitable alpha based on trial-error and plotting iterations vs cost fn
iters=1000        
data.insert(0,'Ones',1)
colsize=data.shape[1]
x=np.matrix(data.iloc[:,:colsize-1])
y=np.matrix(data.iloc[:,colsize-1:colsize])
theta=np.matrix(np.array([0,0]))   
print('shapes of np matrices to be used for matrix multiplication in defined funcs')
print(x.shape,theta.shape,y.shape)  
print('Initial Cost :',CostFunc(x,y,theta))
#print('Cost for each of the 100 iterations',GradDesc(x,y,theta,alpha,100))   ##Cost value should decrease after every iteration 
#perfor linear regression
g,cost=GradDesc(x,y,theta,alpha,iters)
print ('Theta',g)
print ('End cost value after linear regression',CostFunc(x,y,g))

#plot predicted linear regression line
x1 = np.linspace(data.Population.min(), data.Population.max(), 100)  
f = g[0, 0] + (g[0, 1] * x1)
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x1, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')    

#plot cost value with iterations to see cost always decreases
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Cost function vs. Iterations')  


        