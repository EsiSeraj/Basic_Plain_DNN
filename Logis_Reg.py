# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:20:22 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>

# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)

Logistic Regression
    - All Required Helper functions

Reguired Packages
    - numpy
    
# NOTE: this function gets regular updats; for now, it only includes equations
    and computations for sigmoid non-linearity, additional non-linearities are
    to be added in the future
"""

# In[0]: loading packages
import numpy as np

# In[1]: sigmoid function

def sigmoid(z):
    """
    This function Computes the sigmoid of z (corresponding to activation)

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    a -- sigmoid (or activation) of z
    """

    a = 1/(1+np.exp(-z))
    
    return a

# In[2]: parameter initialization
    
def initialize_params(dim, case = "zero" ):
    """
    This function creates a vector of zeros or random values of shape (dim, 1) 
    for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    case -- specify whether you want to initialize with 'zero' or 'random'
    
    Returns:
    w -- initialized weights vector of shape (dim, 1)
    b -- initialized bias scalar
    
    NOTE: unlike DNN, for logistic regression it is OK to initialize with zeros
    
    """
    
    if case == "zero":
        w = np.zeros((dim, 1), dtype=float)
        b = 0
        
    elif case == "random":
        w = np.random.randn(dim, 1)*0.01
        b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# In[3]: forward and backward propagations
    
def propagate(w, b, X, Y):
    """
    This function Implements the cost function (forward) and its gradients
    (backward)

    Arguments:
    w -- weights, a numpy array of size (dim, 1)
    b -- bias, a scalar
    X -- data of size (dim, number of examples)
    Y -- true "label" vector (0/1 binary) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """
    
    m = X.shape[1]
    
    # forward propagation (from X to Cost)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A), axis=1) # cost func
    
    # backward propagation (calculate grads)
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

# In[4]: Optimization

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (dim, 1)
    b -- bias, a scalar
    X -- data of shape (dim, number of examples)
    Y -- true "label" vector (0/1 binary), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        # compute gradiantes
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        
        # update parameters
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the cost value every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

# In[5]: 



