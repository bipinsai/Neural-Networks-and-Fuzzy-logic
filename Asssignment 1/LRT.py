# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:23:13 2019

@author: bipin
"""

#importing all the necessary packages 
import pandas as pd
import numpy as np
import random
import math
#reading the file 
data = pd.read_excel('data3.xlsx', header = None )
data = pd.DataFrame(data)
norma = data.iloc[ :,1:-1]
#NORMALIZING THE FEATURE MATRIX
data.iloc[ :,1:-1] = (norma - norma.mean() ) / norma.std()

# Taking 60 % as training data and 40% as testing data 
train = data.sample(frac=0.6, random_state=random.randint(1,1000))
#droping the training samples from the data dataframe for the testing samples
test = data.drop(train.index)

#Introducing a label for the last column 
train.rename(columns={train.columns[-1]:'y'}, inplace=True)
test.rename(columns={test.columns[-1]:'y'}, inplace=True)

#Taking only the samples containing y = 1 as the y label 
t1 = train.loc[train['y'] == 1]
#Taking only the samples containing y = 1 as the y label 
t2 = train.loc[train['y'] == 2]
#Separating only the features 
train_l1 = t1.iloc[:, :4]
train_l2 = t2.iloc[:, :4]

#Calculating the prior probability for both the training sets 
prior_1 = (train_l1.shape[0])/train.shape[0]
prior_2 = (train_l2.shape[0])/train.shape[0]

#Calculating covariance for the calculation of apostreiori 
covariance_1 = np.cov(train_l1.T)
covariance_2 = np.cov(train_l2.T)

#Training matrix 
X_1 = np.matrix(train_l1.T)
X_2 = np.matrix(train_l2.T)

#calculating mean for the features which will help in the calculation of apostreiori 
mean_train1 = np.array(X_1.mean(1)).flatten()
mean_train2 = np.array(X_2.mean(1)).flatten()

#calculating the denominator for the formula used for the calculation of apostreiori 
denominator_1 = 1/(((2 * math.pi)**2) * ((np.linalg.det(covariance_1))**0.5))
denominator_2 = 1/(((2 * math.pi)**2) * ((np.linalg.det(covariance_2))**0.5))

#taking the test features
test_data_features = test.iloc[:, :4]

#True positives , False positives , true negatives , False negativesa
tp, fp, tn, fn = 0, 0, 0, 0
test_data_features.shape[0]
for i in range(test_data_features.shape[0]):
#    Taking the data point as a row in the test matrix
    test_data_point = np.array(test_data_features.iloc[i, :])
#    Calculating likelihood function value for the y = 1
    likelihood_1 = (denominator_1) * (math.exp((-0.5 * np.matmul(np.matmul((test_data_point - mean_train1), np.linalg.inv(covariance_1)),(test_data_point - mean_train1).T))))
#    Calculating likelihood function value for the y = 1
    likelihood_2 = (denominator_2) * (math.exp((-0.5 * np.matmul(np.matmul((test_data_point - mean_train2), np.linalg.inv(covariance_2)),(test_data_point - mean_train2).T))))
#    calculating the likelihood ratio 
    likelihood_ratio = likelihood_1 / likelihood_2
#   calculating the priori ratio 
    prior_ratio = prior_2/prior_1
#    If likelihood > priori ratio ; IT belongs to class 1 
#    It it is actually 1 ; true positives incerases by one 
#    else false positives increases by one 
#    Same with case of less than symbol 
#    But we use true negatives
    if(likelihood_ratio > prior_ratio):
        if(test.iloc[i, 4] == 1):
            tp += 1
        else:
            fp += 1
    else:
        if(test.iloc[i, 4] == 2):
            tn += 1
        else:
            fn += 1
#Printing the values of the accuracy
print(" Accuracy is    ",(tp+tn)/(tp+fp+tn+fn),"\n Sensitivity is ",tp/(tp+fn)," \n Specificity is ",tn/(tn+fp))
