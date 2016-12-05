import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import os
path=os.getcwd()+'\ex2data1.txt'
data=pd.read_csv(path,header=None,names=['Exam1','Exam2','Admitted'])
pos=data[data['Admitted'].isin([1])]
neg=data[data['Admitted'].isin([0])]
fig,ax=plt.subplots(figsize=(12,8))
ax.scatter(pos.Exam1,pos.Exam2,c='b',s=50,marker='o',label='Admitted')
ax.scatter(neg.Exam1,neg.Exam2,c='r',s=50,marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam1 score')
ax.set_ylabel('Exam2 score')

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
nums = np.arange(-10, 10, step=1)

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(nums, sigmoid(nums), 'r')      

def CostFunc(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    j1=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    j2=np.multiply(1-y,np.log(1-sigmoid(x*theta.T)))
    return np.sum(j1-j2)/(len(x))
    
data.insert(0, 'Ones', 1)
cols=data.shape[1]
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

x=np.array(x.values)
y=np.array(y.values)
theta=np.zeros(3)

print('x,theta,y shapes',x.shape, theta.shape, y.shape)  

print('Initial Cost function value',CostFunc(theta,x,y))

def gradient(theta, x, y):  
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(x * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, x[:,i])
        grad[i] = np.sum(term) / len(x)

    return grad
#Note that we don't actually perform gradient descent in 
#this function - we just compute a single gradient step. 
#In the exercise, an Octave function called "fminunc" is 
#used to optimize the parameters given functions to compute
#the cost and the gradients. Since we're using Python, we 
#can use SciPy's optimization API to do the same thing.

import scipy.optimize as opt  
result = opt.fmin_tnc(func=CostFunc, x0=theta, fprime=gradient, args=(x, y))  
print('Cost value after optimization',CostFunc(result[0], x, y))    

#We now have the optimal model parameters for our data set.
#Next we need to write a function that will output predictions
#for a dataset X using our learned parameters theta. We can then
#use this function to score the training accuracy of our classifier.

def predict(theta, x):  
    probability = sigmoid(x * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])  
predictions = predict(theta_min, x)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print ('accuracy = {0}%'.format(accuracy))  