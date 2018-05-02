# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:20:22 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>
@website: https://github.com/EsiSeraj/

Logistic Regression
    - All required helper functions
    - Sigmoid output for binary classification
    - A comprehensive logistic regression model generator

Reguired Packages
    - numpy
    
# NOTE: this function gets regular updats; for now, it only includes equations
    and computations for sigmoid non-linearity, additional non-linearities are
    to be added in the future
    
Copyright (C) <2018>  <Esmaeil Seraj>
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

"""
# The main steps for building a Neural Network are:
     1. Define the model structure (such as number of input features) 
     2. Initialize the model's parameters
     3. Loop:
         - Calculate current loss (forward propagation)
         - Calculate current gradient (backward propagation)
         - Update parameters (gradient descent)
     4. Use learnt parameters to predict the labels on train data
     5. Use learnt parameters to predict the labels on test data
"""

# In[0]: loading packages
import numpy as np

# In[1]: sigmoid function

def sigmoid(z):
    """
    This function Computes the sigmoid activation of z in numpy

    Arguments:
    z -- A scalar or numpy array of any shape

    Return:
    a -- sigmoid (or activation) of z
    """

    a = 1/(1 + np.exp(-z))
    
    assert(a.shape == z.shape)
    
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

# In[4]: optimization (gradient descent)

def optimize(w, b, X, Y, num_iter, learning_rate = 0.005, print_cost = True):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (dim, 1)
    b -- bias, a scalar
    X -- data of shape (dim, number of examples)
    Y -- true "label" vector (0/1 binary), of shape (1, number of examples)
    num_iter -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule, 
    (default = 0.005)
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with 
            respect to the cost function
    costs -- list of all the costs computed during the optimization, this will 
            be used to plot the learning curve.
    """
    
    costs = []
    
    for i in range(num_iter):     
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

# In[5]: prediction

def predict(w, b, X):
    '''
    This function Predicts whether the label is 0 or 1 using learned logistic 
    regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (dim, 1)
    b -- bias, a scalar
    X -- data of size (dim, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for
    the examples in X
    
    NOTE: the threshold for output probabilities is <th=0.5> where any 
    prediction with a greater value will be labled as '1' and vice versa
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b) # compute probabilities
    
    for i in range(A.shape[1]):     
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        zero_index = np.where(A >= .5, A, 0)
        one_index = np.where(zero_index < .5, zero_index, 1)
        Y_prediction = one_index
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

# In[6]: logistic regression model
    
def logis_reg_model(X_train, Y_train, X_test, Y_test, init_case = "zero", 
                    num_iter = 2000, learning_rate = 0.005, print_cost = True):
    """
    This function Builds a logistic regression model based upon your train data
    
    Arguments:
    X_train -- training set represented by an array of shape (dim, m_train)
    Y_train -- training labels represented by an array of shape (1, m_train)
    X_test -- test set represented by an array of shape (dim, m_test)
    Y_test -- test labels represented by an array of shape (1, m_test)
    num_iter -- hyperparameter: number of iterations for optimization, 
                        (default = 2000)
    learning_rate -- hyperparameter: used in updating stage, (default = 0.005)
    print_cost -- Set to true to print the cost every 100 iterations, (default 
                   = True)
    
    Returns:
    lr_mdl -- dictionary containing information about the model
    
    NOTE: in addition to the model information, this function also prints the
    accuracies for train and test data
    """
    
    # initialize parameters with zeros
    w, b = initialize_params(X_train.shape[0], init_case)

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iter, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train))*100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test))*100))

    
    lr_mdl = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iter": num_iter}
    
    return lr_mdl

# In[]:

