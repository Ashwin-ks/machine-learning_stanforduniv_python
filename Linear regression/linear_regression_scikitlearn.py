# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:05:06 2016

@author: COMP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
model = linear_model.LinearRegression()

path=os.getcwd()+'\ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])


# set X_tr(one feature) and y_tr(target) ------- (training data)
cols = data.shape[1]
#training data
X_tr = data.iloc[:50,0:cols-1]
y_tr = data.iloc[:50,cols-1:cols]
#test data
X_ts = data.iloc[50:,0:cols-1]
y_ts = data.iloc[50:,cols-1:cols]

# Train the model using the training sets
model.fit(X_tr,y_tr)

# Plot outputs
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X_ts,y_ts, label='Traning Data')
ax.plot(X_ts,model.predict(X_ts),'r',label='Prediction')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

print ('Intercept',model.intercept_)
print('Coefficients',model.coef_)
print('Mean squared error\n%.2f'% np.mean((model.predict(X_ts)-y_ts)**2))
# Explained variance score: 1 is perfect prediction
print('Variance score',model.score(X_ts,y_ts))