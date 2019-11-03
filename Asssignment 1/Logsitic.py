# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:57:54 2019

@author: bipin
"""
#importing the necessary packages 
import pandas as pd
import numpy as np
import random

#Sigmoid function for the calculation of the g(z) 
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#reading the file
data = pd.read_excel('data3.xlsx', header = None )

#Adding a colums of ones 
data= np.concatenate((np.ones(shape = (data.shape[0],1)),data) ,axis = 1)
data = pd.DataFrame(data)

#Labelling th data for future use 
data.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'y']

#NORMALIZING THE FEATURE MATRIX
norma = data.iloc[ :,1:-1]
data.iloc[ :,1:-1] = (norma - norma.mean() ) / norma.std()

#initializing the initial theta values with noramlised random values
Theta = np.random.randn(1,5).ravel()
#Taking the learning rate Alpha to be 0.004
alpha = 0.004

# Taking 60 % as training data and 40% as testing data 
train = data.sample(frac=0.6,random_state=random.randint(1,1000))
#droping the training samples from the data dataframe for the testing samples
test = data.drop(train.index)

#Preparing the training data ( separating from the class label )
X = np.array(train.loc[:,['x0', 'x1', 'x2', 'x3', 'x4']])
# Training class data 
Y = np.array(train['y'])
m = Y.shape[0]

#Subtracting to make it binary classes ( 0 or 1 )
for i in range(len(Y)):
    Y[i] -= 1
# Taking the number of iterations to be 700 
for i in range(700):
#    Finding the value of z
    z =np.matmul(X,Theta.T)
#    Putting it in the calculate the value of h(Z)
    h = sigmoid(z)
#    Calculating the gradient 
    gradient = (np.dot(X.T, (h-Y))) / m
#    Updating the Theta values
    Theta-= alpha*gradient

#Calculating the required values for the testing set
X = np.array(test.loc[:,['x0', 'x1', 'x2', 'x3', 'x4']])
Y = np.array(test['y'])
z = np.dot(X, Theta)
h = sigmoid(z)
m = Y.shape[0]

#Converting into binary classes 
for i in range(len(Y)):
    Y[i] -= 1


#True positives , False positives , true negatives , False negatives
tp, fp, tn, fn = 0, 0, 0, 0
for index in range(m):
#    If the h(z) value > 0.5 Belongs to class 1 
#    Else belongs to class 0 
    if (h[index] > 0.5):
        a = 1
    else :
        a= 0
#   Calculating the where it calssified correctly or not 
    if(a == Y[index]):
        if(a == 0):
            tp += 1
        else:
            tn += 1
    else:
        if(a == 0):
            fp += 1
        else:
            fn += 1
#Printing the values
accuracy = (tp+tn) / (tp+tn+fp+fn)
sensitivity = tp / (tp+fn) 
specificity = tn / (tn+fp )
print ( " Accuracy     = " , accuracy , "\n Sensitivity  = " , sensitivity ,"\n Specificity  = ",specificity )