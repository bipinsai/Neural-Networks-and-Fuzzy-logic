# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 02:38:15 2019

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
    #    Making the classes other than wanted to 0 and the wanted to 1
    m = Y.shape[0]
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

def Accuracy(test, theta_1, theta_2, theta_3):
    accuracy, accuracy_1, accuracy_2, accuracy_3 = 0, 0, 0, 0
    m1, m2, m3 = 0, 0, 0
#    Preparing the testing X features
    X = np.array(test.iloc[:, :-1])
#    Preparing the testing Y label 
    Y = np.array(test.iloc[:, -1])
#    Finding the value z
    z1 = np.dot(X, theta_1)
    z2 = np.dot(X, theta_2)
    z3 = np.dot(X, theta_3)
#    Finding the values of h(Z) 
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
#         the training sample belongs to that calss which has the highrst values
#        h value
        ans = max(h1[index], h2[index], h3[index])
        if(ans == h1[index]):
            a = 1
        elif(ans == h2[index]):
            a = 2
        elif(ans == h3[index]):
            a = 3
#            Calculating the accuracies
        if(a == Y[index]):
            accuracy += 1
            if(a == 1):
                accuracy_1 += 1
            elif(a == 2):
                accuracy_2 += 1
            elif(a == 3):
                accuracy_3 += 1
    return (accuracy_1/m1, accuracy_2/m2, accuracy_3/m3, accuracy/m)

#reading the file
data = pd.read_excel('data4.xlsx', header = None )
# making the first as zeroes : 
data= np.concatenate((np.ones(shape = (data.shape[0],1)),data) ,axis = 1)
data = pd.DataFrame(data)

#normalization of data
norma = data.iloc[ :,1:-1]
data.iloc[ :,1:-1] = (norma - norma.mean() ) / norma.std()

#INtitializing random values to theta vector 
Theta = np.random.randn(1,8).ravel()
#taking the learning rate alpha
alpha = 0.01

#Randomizing the dataset 
data = data.sample(frac=1,random_state=random.randint(1,1000))

#THe size of each fold 
sz = int(len(data) * 0.2)

start, end, final_accuracy = 0, 0, 0
for i in range(5):
#    starting index
    start = i*sz
#    ending index 
    end = (i+1)*sz
    if(i == 4):
#        If its the last row do this 
        end = len(data)
#   Separating the train and the test sets 
    train = data.iloc[start:end, :]
    test = data.drop(train.index)
#    Making copies of theta ..
    T1= Theta.copy()
    T2= Theta.copy()
    T3= Theta.copy()
#    making copies of training sets 
    Tr1 = train.copy()
    Tr2 = train.copy()
    Tr3 = train.copy()
#    Finding the values thetas ...
    theta_1 = LogisticRegression(Tr1, T1,  alpha, 1)
    theta_2 = LogisticRegression(Tr2, T2,  alpha, 2)
    theta_3 = LogisticRegression(Tr3, T3,  alpha, 3)
# taking the values of accuracies into these variables 
    a1, a2, a3, a = Accuracy(test, theta_1, theta_2, theta_3)
    final_accuracy += a
# Printing the values of accuracies    
    print("Individual class accuracies for iteration {}: {}, {}, {}".format(i+1, a1, a2, a3))
    print("One vs All overall accuracy for iteration {}: {}".format(i+1, a))
final_accuracy /= 5
print("Average overall accuracy: {}".format(final_accuracy))

























