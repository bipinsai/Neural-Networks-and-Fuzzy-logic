# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:45:32 2019

@author: bipin
"""
#importing the necessary packages 
import pandas as pd
import numpy as np
import random

#Sigmoid function for the calculation of the g(z) 
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def LogisticRegression(train, theta, alpha, class_label):
    #Preparing the training data ( separating from the class label )
    X = np.array(train.iloc[:, :-1])
    # Training class data 
    Y = np.array(train.iloc[:, -1])
    m = Y.shape[0]
#    Making the classes other than wanted to 0 and the wanted to 1
    for i in range(len(Y)):
        if(Y[i] != class_label):
            Y[i] = 0
        else:
            Y[i] = 1
    for i in range(1000):
        #    Finding the value of z
        z = np.dot(X, theta)
        #    Putting it in the calculate the value of h(Z)
        h = sigmoid(z)
        #    Calculating the gradient 
        gradient = np.dot(X.T, (h-Y)) / m
        #    Updating the Theta values
        theta -= alpha * gradient
    return theta


#reading the file
data = pd.read_excel('data4.xlsx', header = None )


#Adding a colums of ones 
data= np.concatenate((np.ones(shape = (data.shape[0],1)),data) ,axis = 1)
data = pd.DataFrame(data)

#NORMALIZING THE FEATURE MATRIX
norma = data.iloc[ :,1:-1]
data.iloc[ :,1:-1] = (norma - norma.mean() ) / norma.std()

#initializing the initial theta values with noramlised random values
Theta = np.random.randn(1,8).ravel()
#Taking the learning rate Alpha to be 0.01
alpha = 0.01

# Taking 60 % as training data and 40% as testing data 

train = data.sample(frac=0.6,random_state=random.randint(1,1000))
# Taking 60 % as training data and 40% as testing data 
test = data.drop(train.index)

#MAking copies of Theta values for the various classufication  models (3)
T1= Theta.copy()
T2= Theta.copy()
T3= Theta.copy()

#MAking copies of Training set  values for the various classufication  models (3)
Tr1 = train.copy()
Tr2 = train.copy()
Tr3 = train.copy()

# Finding the values of theta for the various classufication models 
theta_1 = LogisticRegression(Tr1, T1,  alpha, 1)
theta_2 = LogisticRegression(Tr2, T2,  alpha, 2)
theta_3 = LogisticRegression(Tr3, T3,  alpha, 3)

#Initialising the values of accuracy values
accuracy, accuracy_1, accuracy_2, accuracy_3 = 0, 0, 0, 0
m1, m2, m3 = 0, 0, 0

# preparing the test feature set 
X = np.array(test.iloc[:, :-1])
#Preparing the test class label set
Y = np.array(test.iloc[:, -1])

#Finding the z = g(z) values ...
z1 = np.dot(X, theta_1)
z2 = np.dot(X, theta_2)
z3 = np.dot(X, theta_3)

#Calculating the values of the activation values ...
h1 = sigmoid(z1)
h2 = sigmoid(z2)
h3 = sigmoid(z3)
m = Y.shape[0]

#Calculating the values of no of samples of class 1 and class 2 and class 3 respectively 
for i in range(m):
	if(Y[i] == 1):
		m1 += 1
	elif(Y[i] == 2):
		m2 += 1
	else:
		m3 += 1

#Finding the max values of class functions i.e the training salmle belongs to that class
for index in range(m):
    ans = max(h1[index], h2[index], h3[index])
    if(ans == h1[index]):
        a = 1
    elif(ans == h2[index]):
        a = 2
    elif(ans == h3[index]):
        a = 3
#   Calculating the accuracy values ...
    if(a == Y[index]):
        accuracy += 1
        if(a == 1):
            accuracy_1 += 1
        elif(a == 2):
            accuracy_2 += 1
        elif(a == 3):
            accuracy_3 += 1
# printing the accuracy values 
print( " ONE VS ALL Individual Accuracies :")
print(" ",(accuracy_1/m1) ,"\n ",(accuracy_2/m2) ,"\n ",( accuracy_3/m3))
print ( " Overall Accuracy : " , (accuracy/m))