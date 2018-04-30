# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:20:22 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>
@website: https://github.com/EsiSeraj/

Shalllow Neural Network (one hidden layer) - SNN
    - All required helper functions
    - tanh activation for hidden layer
    - Sigmoid output for binary classification
    - A comprehensive SNN model generator

Reguired Packages
    - numpy
    
# NOTE: this function gets regular updats; for now, it only includes equations
    and computations for sigmoid and ReLU non-linearities, additional 
    non-linearities are to be added in the future
    
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
     2. Initialize the model's parameters and set the hyperparameters
     3. Loop:
         - Calculate current loss (forward propagation)
         - Calculate current gradient (backward propagation)
         - Update parameters (gradient descent)
     4. Use learnt parameters to predict the labels on train data (forward)
     5. Use learnt parameters to predict the labels on test data
"""

# In[0]: loading packages
import numpy as np

# In[1]: adjusting layer dimentions (input - hidden - output)

def layer_sizes(X, Y, n_h):
    """
    This function provides the NN with dimentions of layers
    
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)

# In[2]: sigmoid function

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

# In[3]: parameter initialization
    
def initialize_params(n_x, n_h, n_y):
    """
    This function initializes the parameters w and b for NN
    
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing model's initial parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# In[4]: forward propagation
    
def forward_propagation(X, parameters):
    """
    This function implements the forward path for NN
    
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of 
    "initialize_params()")
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# In[5]: compute cost function
    
def cross_ent_cost(A2, Y, parameters):
    """
    This function Computes the cross-entropy cost function
    
    Arguments:
    A2 -- The sigmoid output of the second activation (output layer), of shape 
    (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing all parameters of NN
    
    Returns:
    cost -- cross-entropy cost 
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -(1/m)*np.sum(logprobs)
    cost = np.squeeze(cost)
    
    assert(isinstance(cost, float))
    
    return cost

# In[6]: backward propagation
    
def backward_propagation(parameters, cache, X, Y):
    """
    This function Implements the backward propagation
    
    Arguments:
    parameters -- python dictionary containing all model parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (dim, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to parameters
    """
    m = X.shape[1]
    
#    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# In[7]: updating parameters
    
def update_parameters(parameters, grads, learning_rate = 0.5):
    """
    This function updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing all model parameters 
    grads -- python dictionary containing computed gradients
    learning_rate -- learning rate of the gradient descent update rule, 
    (default = 0.5)
    
    Returns:
    parameters -- python dictionary containing all updated parameters 
    """
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# In[8]: shallow neural network (one hidden layer) (SNN) model
    
def shallow_nn_model(X, Y, n_h, num_iterations = 10000, learning_rate = 0.5, 
                     print_cost = True):
    """
    This function Builds a shallow neural network (one hidden layer)(SNN) model
    based upon your train data
    
    Arguments:
    X -- dataset of shape (dim, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer (# hidden units)
    num_iterations -- Number of iterations in gradient descent loop
    learning_rate -- learning rate of the gradient descent update rule, 
    (default = 0.5)
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model by which prediction can occur
    costs -- list of all the costs computed during the optimization, this will 
            be used to plot the learning curve.
    """

    n_x = layer_sizes(X, Y, n_h)[0]
    n_y = layer_sizes(X, Y, n_h)[2]
    
    parameters = initialize_params(n_x, n_h, n_y)
#    W1 = parameters["W1"]
#    b1 = parameters["b1"]
#    W2 = parameters["W2"]
#    b2 = parameters["b2"]
    
    costs = []
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function
        cost = cross_ent_cost(A2, Y, parameters)
 
        # Backpropagation
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Record the cost value every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters, costs
    
# In[9]: prediction
    
def predict(parameters, X):
    """
    This function predicts a class for each example in X using the built model
    
    Arguments:
    parameters -- python dictionary containing all model parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (0/1 binary)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 
    # using 0.5 as the threshold
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

# In[]: 


