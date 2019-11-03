# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:41:03 2019

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

# Building n * (n-1 ) / 2 models 
# A : Calssifier for 1,2 classification 
# B : Classifier for 2 , 3 classification
# C: Classifier for 1, 3 classification
A, B, C = [1, 2], [2, 3], [1, 3]
#Renaming the last label as y for easiness
train.rename(columns={train.columns[-1]:'label'}, inplace=True)
# Separating all the samples form the training dat abelonging to class 1,2
Tr1 = train.loc[train['label'].isin(A)]
# Separating all the samples form the training dat abelonging to class 2,3
Tr2 = train.loc[train['label'].isin(B)]
# Separating all the samples form the training dat abelonging to class 1,3
Tr3 = train.loc[train['label'].isin(C)]

#Finding the theta values for the three models developed 
theta_1 = LogisticRegression(Tr1, T1,  alpha, 1)
theta_2 = LogisticRegression(Tr2, T2,  alpha, 2)
theta_3 = LogisticRegression(Tr3, T3,  alpha, 3)

#Initializing the accuracy values
accuracy, accuracy_1, accuracy_2, accuracy_3 = 0, 0, 0, 0
#For counting the no of test samples which belong to a particlualr class
m1, m2, m3 = 0, 0, 0 
#Preparing the testing set
X = np.array(test.iloc[:, :-1])
#preparing the testing label 
Y = np.array(test.iloc[:, -1])
# The z values for all the models developed
z1 = np.dot(X, theta_1)
z2 = np.dot(X, theta_2)
z3 = np.dot(X, theta_3)
#The hypothesis values for the various models
h1 = sigmoid(z1)
h2 = sigmoid(z2)
h3 = sigmoid(z3)
m = Y.shape[0]
#Finding the class which the testing samples belong to : 
for i in range(m):
	if(Y[i] == 1):
		m1 += 1
	elif(Y[i] == 2):
		m2 += 1
	else:
		m3 += 1
for index in range(m):
    c1, c2, c3 = 0, 0, 0
#    > 0.5 belongs to class 1 else belongs to class 2
    a = 1 if h1[index] > 0.5 else 0
#    > 0.5 belongs to class 2 else belongs to class 3
    b = 1 if h2[index] > 0.5 else 0
#    > 0.5 belongs to class 1 else belongs to class 3
    c = 1 if h3[index] > 0.5 else 0
    c1 += (a == 1)
    c1 += (c == 0)
    c2 += (a == 0)
    c2 += (b == 1)
    c3 += (b == 0)
    c3 += (c == 1)
#    basically finding the mode value so that we can decide to which class the testing 
#    sample belongs to 
    max_class = max(c1, c2, c3)
    if(c1 == max_class):
        a = 1
    elif(c2 == max_class):
        a = 2
    elif(c3 == max_class):
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

#Printing the values of the accuracies : 
print( " ONE VS ALL Individual Accuracies :")
print(" ",(accuracy_1/m1) ,"\n ", (accuracy_2/m2) ,"\n ",( accuracy_3/m3))
print ( " Overall Accuracy : " , (accuracy/m))




