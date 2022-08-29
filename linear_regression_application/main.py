'''generating cost function to find (J) 
   J=1/2m * sum (((theta**T * x ) - y)**2)'''
def generate_cost(X,Y,THETA):
    inner = np.power(((X * THETA.T ) - Y) , 2)
    J = np.sum(inner)/(2*len(X))
    return J


'''
creating the gradient descent function to generate 
thetas that bring the lowest cost (J) based on alpha 
and number of iterations 
'''
def gradient_descent(X,Y,THETA,ALPHA,ITERS):
    THETAS = np.matrix(np.zeros(THETA.shape))
    N_THETA = int(THETA.ravel().shape[1])
    n_costs = np.zeros(ITERS)
    for i in range(ITERS):
        error = (X * THETA.T)-Y
        
        for THETA_N in range (N_THETA):
            term = np.multiply(error , X[:,THETA_N])
            THETAS[0,THETA_N] = THETA[0,THETA_N] - ((ALPHA/len(X))*np.sum(term))
            
            # THETAS[0,THETA_N] = THETA[0,THETA_N] - (ALPHA * generate_cost(X,Y,THETA))
    
        THETA = THETAS
        n_costs[i]=generate_cost(X, Y, THETA)
    
    return THETA , n_costs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#READING DATA
df=pd.read_csv('data.txt' , header=None , names=['size','rooms','price'])
#print(df.head(10))

'''
RESCALING THE DATA TO BE SPREAD WELL
(OBSERVATION - THE AVERAGE ) / RANGE OR STANDERD DEVIATION
SO ALL NUMBERS WILL RANGE FROM 1 TO -1
''' 

df = (df - df.mean()) / df.std()
#print(df.head())

#ADD ONES COLUMN FOR MATRIX MULTIPLICATION
df.insert(0,'ones',1)

#seperate training data from target value 
cols = df.shape[1]
x = df.iloc[:,0:cols-1]
y = df.iloc[:,cols-1:cols]

#print(x.sample())
#print(y.sample())

#converting dataframes into matrices
x = np.matrix(x.values)
y = np.matrix(y.values)

#initialize theta 
theta = np.matrix(np.array([0,0,0]))


#initializing learning rate and iterations for the gradient descent function
alpha = 0.001
iters = 2000

# print(theta.T.shape)
# print(x.shape)

# cost = generate_cost(x,y,theta)
# print(cost)

best_thetas , gradual_costs = gradient_descent(x,y,theta,alpha,iters)

# print(gradual_costs)
# print('...........................')
# print(best_thetas)

'''graph minimizing cost per each iteration
 so we can define the approximate ideal number of iterations'''
fig , ax = plt.subplots(figsize = (5,5))
ax.plot(np.arange(iters) ,gradual_costs)
plt.show()

'''making formulas of two fit lines 
the first is for size and price and 
the second is for bedrooms and price , 
to use them in plotting'''

x_s = np.linspace(df['size'].min(), df['size'].max(), 100)
size_line = best_thetas[0, 0] + (best_thetas[0, 1] * x_s)
print(size_line)

x_r = np.linspace(df['rooms'].min(), df['rooms'].max(), 100)
rooms_line = best_thetas[0, 0] + (best_thetas[0, 1] * x_r)
print(rooms_line)


# plotting size and price graph
fig , ax = plt.subplots(figsize=(5,5))
ax.plot(x_s , size_line )
ax.scatter(df['size'] , df['price'])
ax.set_xlabel('size')
ax.set_ylabel('price')
ax.set_title('relation between size and price')
plt.show()

# plotting rooms and price graph
fig , ax = plt.subplots(figsize=(5,5))
ax.plot(x_r , size_line )
ax.scatter(df['rooms'] , df['price'])
ax.set_xlabel('rooms')
ax.set_ylabel('price')
ax.set_title('relation between rooms and price')
plt.show()









