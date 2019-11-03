# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:38:08 2019

@author: bipin
"""

#importing all the necessary packages 
import pandas as pd
import numpy as np
import random
import math
#reading the file 
data = pd.read_excel('data4.xlsx', header = None )
#NORMALIZING THE FEATURE MATRIX
norma = data.iloc[ :,1:-1]
data.iloc[ :,1:-1] = (norma - norma.mean() ) / norma.std()

# Taking 70 % as training data and 30% as testing data 
train = data.sample(frac=0.7, random_state=random.randint(1,1000))
#droping the training samples from the data dataframe for the testing samples
test = data.drop(train.index)

#Introducing a label for the last column of both training set and testing set 
train.rename(columns={train.columns[-1]:'y'}, inplace=True)
test.rename(columns={test.columns[-1]:'y'}, inplace=True)

#Taking only the samples containing y = 1 y =2 y = 3 as the y label 
#respectively 
t1 = train.loc[train['y'] == 1]
t2 = train.loc[train['y'] == 2]
t3 = train.loc[train['y'] == 3]

#Separating only the features 
train_l1 = t1.iloc[:, :7]
train_l2 = t2.iloc[:, :7]
train_l3 = t3.iloc[:, :7]

#Separating the tesing feature as well 
t1 = test.loc[test['y'] == 1]
t2 = test.loc[test['y'] == 2]
t3 = test.loc[test['y'] == 3]

test_l1 = t1.iloc[:, :7]
test_l2 = t2.iloc[:, :7]
test_l3 = t3.iloc[:, :7]

#Calculating the prior probability for both the training sets 
prior_1 = train_l1.shape[0]/train.shape[0]
prior_2 = train_l2.shape[0]/train.shape[0]
prior_3 = train_l3.shape[0]/train.shape[0]

#Calculating covariance for the calculation of apostreiori 
covariance_1 = np.cov(train_l1.T)
covariance_2 = np.cov(train_l3.T)
covariance_3 = np.cov(train_l3.T)

#Training matrix 
X_1 = np.matrix(train_l1.T)
X_2 = np.matrix(train_l2.T)
X_3 = np.matrix(train_l3.T)

#calculating mean for the features which will help in the 
#calculation of apostreiori 
mean_train1 = np.array(X_1.mean(1)).flatten()
mean_train2 = np.array(X_2.mean(1)).flatten()
mean_train3 = np.array(X_3.mean(1)).flatten()

#calculating the denominator for the formula used for 
#the calculation of apostreiori 
denominator_1 = 1/(((2 * math.pi)**3.5) * ((np.linalg.det(covariance_1))**0.5))
denominator_2 = 1/(((2 * math.pi)**3.5) * ((np.linalg.det(covariance_2))**0.5))
denominator_3 = 1/(((2 * math.pi)**3.5) * ((np.linalg.det(covariance_3))**0.5))

#taking the test features
test_data_features = test.iloc[:, :7]

#Initializing the accuracies values 
acc, acc1, acc2, acc3 = 0, 0, 0, 0

for i in range(test.shape[0]):
    #    Taking the data point as a row in the test matrix
    test_data_point = np.array(test_data_features.iloc[i, :])
#    Calculating the aposteririori for the 3 classes 
    aposteriori_1 = denominator_1 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train1), np.linalg.inv(covariance_1)),(test_data_point - mean_train1)))
    aposteriori_2 = denominator_2 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train2), np.linalg.inv(covariance_2)),(test_data_point - mean_train2)))
    aposteriori_3 = denominator_3 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train3), np.linalg.inv(covariance_3)),(test_data_point - mean_train3)))
# Multiplying with prior probabilty to get p ( yk / X )  
    aposteriori_1 = aposteriori_1*prior_1
    aposteriori_2 = aposteriori_2*prior_2
    aposteriori_3 = aposteriori_3*prior_3
#    The training sample belongs to the class which has the maximum value
    max_ap = max(aposteriori_1, aposteriori_2, aposteriori_3) 
    if(max_ap == aposteriori_1):
        ans= 1
    elif(max_ap == aposteriori_2):
        ans = 2
    else : 
        ans =3 
#    Calculating the accuracies
    if(ans == test.iloc[i, 7]):
        acc += 1
        if(ans == 1):
            acc1 += 1
        elif(ans == 2):
            acc2 += 1
        else:
            acc3 += 1
#printing the accuracies : 
            
print ( " Overall Accuracy : ", acc/test.shape[0] )
print ( " Class 1 Accuracy : ", acc1/test_l1.shape[0] )
print ( "Class 2 Accuracy : ", acc2/test_l2.shape[0] )
print ( " Class 3 Accuracy : ", acc3/test_l3.shape[0] )

    
    