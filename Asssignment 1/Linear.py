#importing necessary packages 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dataframe = pd.read_excel('data.xlsx', header = None )
# Getting the X1 and x2 matrix 
X = dataframe.iloc[:, 0:2].values
#NORMALIZING THE FEATURE MATRIX
normal_first_column =  (X[:,0] - np.mean(X[:,0]))/np.std(X[:,0])
normal_second_column = (X[:,1] - np.mean(X[:,1]))/np.std(X[:,1])
X[:,0] = normal_first_column
X[:,1] = normal_second_column
# Appending the X0 feature with all  ones
X= np.concatenate((np.ones(shape = (X.shape[0],1)),X) ,axis = 1)
# Getting the Y values
Y = dataframe.iloc[:,2:3].values
# Setting the learning rate alpha after trail and error on its value and 
#looking at the graph plotted
alpha = 0.0001
# Taking random integers as the intial Theta Matrix  
Theta = np.random.randn(3,1)

# Variable for the J (cost function values ) during various iterations

##############################################Linear Regresion ###################################
J_of_theata = []
# creating copies of sameTheta so that we comapre in the end
Theta_Linear = Theta.copy()
Theta_ridge = Theta.copy()
Theta_least = Theta.copy()
Theta_New = Theta.copy()


#
#fixing the number of iterations to be 100 
for iteration in range(100):
#    Calculating the Hypothesis value 
    H = (np.matmul(X,Theta_Linear)-Y)
#    Calculating the new theta values
    for j in range(3):
        Theta_New[j][0] = Theta_Linear[j][0] - (alpha * np.matmul(H.T,X[:,j]))
#    Updating the Theta values
    for j in range(3):
        Theta_Linear[j][0] = Theta_New[j][0]
#        Calculating the cost value
    J =(sum( np.square(H))*0.5)/X.shape[0]
#    Appending to a list all the J vallues for a partivular iteration
    J_of_theata.append(J)

#Printing the values of theta and final cost 
print ( " Batch gradient Theta values : " )
print( np.array(Theta_Linear))
print(  " Final Cost : ", J_of_theata[-1] )

#Plotting the cost vs Iterations graph

plt.plot(J_of_theata)
plt.show()

    
# Creating a list for the x and y axis  for the 3d plot 
x_axis = np.ndarray(shape= (100,100))
y_axis = np.ndarray(shape = (100,100))

w1 = []
w2 = []


J1 = np.ndarray(shape = (100,100))
  

#Initialising min and max values
min_w1 = -10
max_w1 = 13
min_w2 = -10
max_w2 = 10

#Filling w1 and w2 values from min to max at regukar intervals
continuous_w1 = np.linspace( min_w1,max_w1,100)
continuous_w2 = np.linspace( min_w2,max_w2,100)

for i in range (100):
    for j in range (100):
        h =0
        for k in range (X.shape[0]):
#           Finding the hypothesis value of all values of pair ( w1,W2 )...
            h+= (continuous_w1[i]*X[k][1]+continuous_w2[j]*X[k][2]-Y[k][0])**2
#            Initialising the a_axis and Y axis values .....
        x_axis[i][j] = continuous_w1[i];
        y_axis[i][j] = continuous_w2[j];
#        Finding the cost value for the hypothesis
        J1[i][j] = (( h*0.5)/X.shape[0])

ax = plt.axes(projection='3d')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('COST')
ax.plot_surface(x_axis,y_axis,J1,cmap = "coolwarm")
plt.show()



# Ridge Regression 

# Taking the regularization parameter to be 0.04
lamb = 0.005
J_of_ridge = []
Theta_ridge = Theta.copy()

for iteration in range(100):
#   CAlculating the value of the hypothesis 
    H = (np.matmul(X,Theta_ridge)-Y)
#    Calculating the new theta values
    for j in range(3):
        Theta_New[j][0] = (1- (alpha* lamb ))* Theta_ridge[j][0] - (alpha * np.matmul(H.T,X[:,j]))
#   Updating the value of theta  
    for j in range(3):
        Theta_ridge[j][0] = Theta_New[j][0]
    J =(sum( np.square(H))*0.5)/X.shape[0] + lamb * np.sum(np.square(Theta_ridge))*0.5
    J_of_ridge.append(J)
    
# Printing the values of theta and final cost : 
print ( " Ridge Theta Values :" )
print ( Theta_ridge)
print ( "Final Cost  " )
print ( J_of_ridge[-1] )

# PLotting the values 

plt.plot(J_of_ridge)
plt.show()

    

min_w1 = -10
max_w1 = 13
min_w2 = -10
max_w2 = 10

x_axis = np.ndarray(shape= (100,100))
y_axis = np.ndarray(shape = (100,100))

J_ridge = np.ndarray(shape = (100,100))

w1 = []
w2 = []
J_of_ridge = []
# 
continuous_w1 = np.linspace( min_w1,max_w1,100)
continuous_w2 = np.linspace( min_w2,max_w2,100)

for i in range (100):
    for j in range (100):
        h =0
        for k in range (X.shape[0]):
            h+= (continuous_w1[i]*X[k][1]+continuous_w2[j]*X[k][2]-Y[k][0])**2 
        x_axis[i][j] = continuous_w1[i];
        y_axis[i][j] = continuous_w2[j];
        J_ridge[i][j] = (( h*0.5)/X.shape[0]) + lamb * ((continuous_w1[i]**2 )+(continuous_w2[j]**2))*0.5



ax = plt.axes(projection='3d')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost')
ax.plot_surface(x_axis,y_axis,J1,cmap = "coolwarm")
plt.show()



#Least Angle regression 

# Creating copies of Theta values
Theta_least = Theta.copy()
Theta_New = Theta.copy()

# Assumong the regularization parameter 
lamb = 0.2
# Assuming the learning rate alpha 
alpha  = 0.0005
J_of_least=[]
for iteration in range(100):
    H = (np.matmul(X,Theta_least)-Y)
    for j in range(3):
#        Findinf the values of new Theta values
        Theta_New[j][0] = Theta_least[j][0] - (alpha * np.matmul(H.T,X[:,j])) -  alpha*lamb*  np.sign(Theta_least[j][0]) * 0.5
    for j in range(3):
#       Updating the values of theta 
        Theta_least[j][0] = Theta_New[j][0]
#        Calculating the cost values 
    J =(sum( np.square(H))*0.5)/X.shape[0] + lamb * np.sum(np.abs(Theta_least))*0.5
    J_of_least.append(J)

#Printing and plotting the values : 
print ( " The theta values using LEast angle regression using Batch gradient descent")
print(Theta_least)
print( " Final Cost: " )
print( J_of_least[-1])

plt.plot(J_of_least)
plt.show()


###################################################FINISHED LINEAR REGRESSION #######################

# Taking random integers as the intial Theta Matrix  
Theta_Stochastic = Theta.copy()
Theta_ridge = Theta.copy()
Theta_least =  Theta.copy()
Theta_New = Theta.copy()

# Taking the learning rate Alpha : 
alpha = 0.08
#Tqking the value of lamda 
# List to contain the values of Stochatic_J
Stochastic_J = []
# Calculating the intial value of H : 
H = (np.matmul(X,Theta_Stochastic)-Y)
for iteration in range ( 100 ):
#    Taking a random samples from the set 
    i = random.randint(0,X.shape[0]-1)
#    Finding the new values of Theata Stochastic
    for j in range(3):
        Theta_New[j][0] = Theta_Stochastic[j][0] - ( alpha * H[i][0] * X[i][j] )
#        Updating the values of Theta Stochastic
    for j in range(3):
        Theta_Stochastic[j][0] = Theta_New[j][0]
#        Finding the updated H value
    H = (np.matmul(X,Theta_Stochastic)-Y)
#    Finding the Cost values
    Jk =(sum( np.square(H))*0.5)/X.shape[0]
    Stochastic_J.append(Jk)

print (" Stochastic Gradient Theta Values : " )
print ( Theta_Stochastic)
print (" Final Cost :" )
print (Stochastic_J[-1] )

plt.plot(Stochastic_J)
plt.show(Stochastic_J)

# Creating a list for the x and y axis  for the 3d plot 
x_axis = np.ndarray(shape= (100,100))
y_axis = np.ndarray(shape = (100,100))

J1 = np.ndarray(shape = (100,100))

w1 = []
w2 = []


    
min_w1 = -10
max_w1 = 13
min_w2 = -10
max_w2 = 10
 
continuous_w1 = np.linspace( min_w1,max_w1,100)
continuous_w2 = np.linspace( min_w2,max_w2,100)

for i in range (100):
    for j in range (100):
        h =0
        for k in range (X.shape[0]):
            h+= (continuous_w1[i]*X[k][1]+continuous_w2[j]*X[k][2]-Y[k][0])**2
        x_axis[i][j] = continuous_w1[i];
        y_axis[i][j] = continuous_w2[j];
        J1[i][j] = (( h*0.5)/X.shape[0])
    
ax = plt.axes(projection='3d')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost')
ax.plot_surface(x_axis,y_axis,J1,cmap = "coolwarm")
plt.show()




#########################
print (Stochastic_J[-1])

print (Theta_Stochastic)

# Ridge Regression 
#########################

x_axis = np.ndarray(shape= (100,100))
y_axis = np.ndarray(shape = (100,100))

J1 = np.ndarray(shape = (100,100))

w1 = []
w2 = []


Theta_ridge = Theta.copy()
#Storing the values of codt for every iteration : 
StochasticRidge_J = []
# Calculating the intial value of H : 
H = (np.matmul(X,Theta_ridge)-Y)
# Taking the learning rate Alpha : 
alpha = 0.04
#Taking the value of lamba 
lamb = 0.6
for iteration in range ( 1000 ):
#    Taking a random samples from the set 
    i = random.randint(0,X.shape[0]-1)
#    Finding the new values of Theata Stochastic Using regularization 
    for j in range(3):
        Theta_New[j][0] = (1- (alpha* lamb ))* Theta_ridge[j][0] - ( alpha * H[i][0] * X[i][j] )
#        Updating the vallues of Theta Stochastic Ridge 
    for j in range(3):
        Theta_ridge[j][0] = Theta_New[j][0]
#        Finding the values of H 
    H = (np.matmul(X,Theta_ridge)-Y)
#    Caluclation the cost function using the formula
    Jk =((sum( np.square(H))*0.5)/X.shape[0]) + (lamb * np.sum(np.square(Theta_ridge))*0.5)
    StochasticRidge_J.append(Jk)
    
#    Printing the values of final cost and theta values found
print ( " Ridge Theta  (using Stochastic Gradient) Values :" )
print ( Theta_ridge)
print ( "Final Cost  " )
print (  StochasticRidge_J[-1] )

#Plotting the graphs 
plt.plot(StochasticRidge_J)
plt.show()
    
min_w1 = -10
max_w1 = 13
min_w2 = -10
max_w2 = 10
 
continuous_w1 = np.linspace( min_w1,max_w1,100)
continuous_w2 = np.linspace( min_w2,max_w2,100)

for i in range (100):
    for j in range (100):
        h =0
        for k in range (X.shape[0]):
#           Finding the hypothesis value of all values of pair ( w1,W2 )...
            h+= (continuous_w1[i]*X[k][1]+continuous_w2[j]*X[k][2]-Y[k][0])**2
#            Initialising the a_axis and Y axis values .....
        x_axis[i][j] = continuous_w1[i];
        y_axis[i][j] = continuous_w2[j];
        J1[i][j] = (( h*0.5)/X.shape[0])+ lamb * ((continuous_w1[i]**2 )+(continuous_w2[j]**2))*0.5
   
    
ax = plt.axes(projection='3d')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost')
ax.plot_surface(x_axis,y_axis,J1,cmap = "coolwarm")
plt.show()


#Least angle 
Theta_least =  Theta.copy()
Least_angle_sto = []
# Assumong the regularization parameter 
lamb = 0.09
# Assuming the learning rate alpha 
alpha  = 0.05
# Finding the initial value of H 
H = (np.matmul(X,Theta_Stochastic)-Y)
for iteration in range ( 100 ):
#    Taking random samples form the dataset 
    i = random.randint(0,X.shape[0]-1)
    for j in range(3):
#        Findinf the values of new Theta values
        Theta_New[j][0] = Theta_least[j][0] - ( alpha * H[i][0] * X[i][j] ) - alpha* lamb*  np.sign(Theta_least[j][0]) * 0.5
    for j in range(3):
#       Updating the values of theta 
        Theta_least[j][0] = Theta_New[j][0]
#        Calculating the value of h 
    H = (np.matmul(X,Theta_least)-Y)
#        Calculating the cost value
    Jk =(sum( np.square(H))*0.5)/X.shape[0] + lamb * np.sum(np.abs(Theta_least))*0.5
    Least_angle_sto.append(Jk)
    
#Printing the value of theta values found
print( "The theta values using LEast angle regression using Stochastic gradient descent")
print(Theta_least)
print( " Final value of cost " )
print( Least_angle_sto[-1])
plt.plot(Least_angle_sto)
plt.show()


#########################Vectorised Linear Regression ######################

# The theta values for Vectorised Linear Regression using the formula 
Final_O_Vectorisation = np.matmul(np.matmul(np.linalg.inv( np.matmul( X.T , X)) , X.T),Y)
#Finding ( Y - WX ) aaccording to the formula 
K = Y - np.matmul(X,Final_O_Vectorisation )
#FInding its transpose 
J2 = K.T
#Calculating the cost function 
J_of_vectorisation = (np.matmul ( J2,K) *0.5)/X.shape[0]

# Printing the values for comaprison
print(" Cost found through Vectorization :")
print( J_of_vectorisation )
print( " Cost Found through Batch Gradient descent : ")
print (J_of_theata[-1])
print (" Cost Found through Stochastic Gradient descent : " )
print ( Stochastic_J[-1])
print ( " Theta Values found Through vectorization ")
print ( Final_O_Vectorisation)
print ( " Theta Values found Through Batch Gradient descent :  ")
print ( Theta_Linear)
print ( " Theta Values found Through Stochastic Gradient descent :  ")
print( Theta_Stochastic)

############################################################################