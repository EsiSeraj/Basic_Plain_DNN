# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:20:22 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>
@website: https://github.com/EsiSeraj/

Plain Deep Neural Network (Without any improvement technique, a plain 
                           implementation of deep neural networks)
    - All required helper functions
    - tanh, relu and sigmoid activations for hidden layers
    - Sigmoid output for binary classification
    - A comprehensive but plain DNN model generator

Reguired Packages
    - numpy
    - matplotlib.pyplot
    
# NOTE: this function gets regular updats; for now, it only includes equations
    and computations for sigmoid, tanh and ReLU non-linearities, additional 
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
import matplotlib.pyplot as plt

# In[1]: activation functions

def sigmoid(Z):
    """
    This function Computes the sigmoid activation of z in numpy
    
    Arguments:
    Z -- A scalar or numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1 + np.exp(-Z))
    cache = Z
    
    assert(A.shape == Z.shape)
    
    return A, cache

def relu(Z):
    """
    This function implements the RELU function in numoy
    
    Arguments:
    Z -- A scalar or numpy array of any shape
    
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = np.maximum(0, Z)
    cache = Z 
    
    assert(A.shape == Z.shape)
    
    return A, cache

def tanh(Z):
    """
    This function implements (uses) the tanh function in numoy for hidden-layer
    activation function
    
    Arguments:
    Z -- A scalar or numpy array of any shape
    
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = np.tanh(Z)
    cache = Z 
    
    assert(A.shape == Z.shape)
    
    return A, cache

def sigmoid_backward(dA, cache):
    """
    This function implements the backward propagation for a single SIGMOID unit
    
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation
    
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu_backward(dA, cache):
    """
    This function implements the backward propagation for a single RELU unit
    
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation
    
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy = True) # just converting dz to a correct object
    
    # setting dz = 0, When z <= 0
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def tanh_backward(dA, cache):
    """
    This function implements the backward propagation for a single tanh unit
    
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation
    
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = np.tanh(Z)
    dZ = dA * (1 - np.power(s, 2))
    
    assert (dZ.shape == Z.shape)
    
    return dZ

# In[2]: initialize parameters
    
def initialize_parameters_deep(layer_dims):
    """
    This function initializes the parameters w and b for an L-layer NN
    
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer 
    in our network
    
    Returns:
    parameters -- python dictionary containing all parameters "W1..l", "b1..l",
                    where:
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
    
# In[3]: forward propagation module
    
def linear_forward(A_prev, W, b):
    """
    This function implements the linear part of a layer's forward propagation

    Arguments:
    A_prev -- activations from previous layer (or input data) of shape: 
        (dimension of previous layer, number of examples)
    W -- weights matrix: numpy array of shape:
        (dimension of current layer, size of previous layer)
    b -- bias vector, numpy array of shape: 
        (dimension of the current layer, 1)

    Returns:
    Z -- pre-activation, the input of the activation function (non-linear part)
    cache -- a python dictionary containing "A", "W" and "b"; stored for 
            computing the backward pass efficiently
    """
    
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
    return Z, cache

def activation_forward(A_prev, W, b, activation):
    """
    This function implements the activation function applied to the linear 
    forward (pre-activation)

    Arguments:
    A_prev -- activations from previous layer (or input data) of shape: 
                (dimension of previous layer, number of examples)
    W -- weights matrix: numpy array of shape:
            (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape:
            (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text 
                    string: "sigmoid", "relu" or "tanh"

    Returns:
    A -- post-activation, the output of the activation function
    cache -- python dictionary containing "linear_cache" & "activation_cache";
             stored for computing the backward pass efficiently
             just remember that, linear_cache contains "activation from 
             previous layer" + "W" + "b" and activation_cache contains "Z", all
             corresponding to the same block (layer)
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation_deep(X, parameters, activation):
    """
    This function implements forward propagation for an L-layer deep NN with
    sigmoid output layer
    
    Arguments:
    X -- data, numpy array of shape (input dimension, number of examples)
    parameters -- output of initialize_parameters_deep(), containing all model 
                    parameters
    activation -- the activation function to be used in forward path, stored
                    as a string "sigmoid", "tanh", "relu"
    
    Returns:
    AL -- last post-activation value (or y_hat, a.k.a probabilities)
    caches -- list of caches containing every cache of activation_forward() 
                (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network
    
    # forward path from input to layer L-1
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W' + str(l)], 
                                     parameters['b' + str(l)], activation)
        caches.append(cache)
            
    # last part of the forward path (L-th layer), or output layer
    AL, cache = activation_forward(A, parameters['W' + str(L)], 
                                 parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

# In[4]: cost function
    
def compute_cost(AL, Y):
    """
    This function Computes the cross-entropy cost function

    Arguments:
    AL -- probability vector corresponding to all label predictions of shape:
            (1, number of examples)
    Y -- "true" binary labels vector (i.e. containing 0 if non-cat, 1 if cat)
            of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = -(1/m)*np.sum((np.multiply(Y, np.log(AL))) + (np.multiply(1-Y, np.log(1-AL))))
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    
    return cost

# In[5]: backward propagation module
    
def linear_backward(dZ, cache):
    """
    This function implements the linear portion of backward propagation for a 
    single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output 
            (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation
                in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation 
                (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), 
            same shape as W
    db -- Gradient of the cost with respect to b (current layer l), 
            same shape as b
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def activation_backward(dA, cache, activation):
    """
    This function implements the backward propagation for the activation 
    function applied to the linear forward during forward pass
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) that we stored in 
                forward path
    activation -- the activation that was used in this layer during forward 
                    path, stored as a text string: "sigmoid", "relu" or "tanh"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation
                (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l),
            same shape as W
    db -- Gradient of the cost with respect to b (current layer l),
            same shape as b
    """
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def backward_propagation_deep(AL, Y, caches, activation):
    """
    This function implements the backward propagation for an L-layer deep NN 
    with sigmoid output layer
    
    Arguments:
    AL -- probability vector, output of the forward propagation
    Y -- "true" binary labels vector (i.e. containing 0 if non-cat, 1 if cat)
    caches -- list of caches we stored in forward path
    activation -- the activation function that was used in forward path, stored
                    as a string "sigmoid", "tanh", "relu"
    
    Returns:
    grads -- A dictionary with all the gradients, including: "dA", "dW" & "db" 
    """
    
    grads = {}
    L = len(caches) # the number of layers
#    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # just to make sure!
    
    # Initializing the backpropagation with derivative of cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1] # corresponding to the last layer
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, 
                                                              current_cache, activation = "sigmoid")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l+1)], 
                                                           current_cache, activation)
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# In[6]: parameter update
    
def update_parameters_deep(parameters, grads, learning_rate = 0.005):
    """
    This function updates parameters in an L-layer DNN using gradient descent
    
    Arguments:
    parameters -- python dictionary containing all model parameters 
    grads -- python dictionary containing all gradients
    learning_rate -- learning rate of the gradient descent update rule, 
                        (default = 0.005)
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

    return parameters

# In[7]: Plain deep neural network model generator
    
def plain_nn_model_deep(X, Y, layer_dims, activation, learning_rate = 0.005, 
                        num_iterations = 3000, print_cost = True, plot_lrn_curve = True):
    """
    This function implements an L-layer deep neural network in the format of: 
    ["linear"->"activation"]*(L-1) --> "linear"->"sigmoid"
    
    Arguments:
    X -- data, numpy array of shape (dimension, number of examples)
    Y -- "true" binary labels vector (i.e. 0 if cat, 1 if non-cat), of shape: 
        (1, number of examples)
    layer_dims -- list containing the input and each other layer's dimension, 
                    of length (number of layers + 1)
    activation -- the activation function to be used in forward path, stored
                    as a string "sigmoid", "tanh", "relu"
    learning_rate -- learning rate of the gradient descent update rule, 
                        (default = 0.005)
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    plot_lrn_curve -- if True, it plots the learning curve
    
    Returns:
    parameters -- parameters learnt by the model by which prediction can occur
    costs -- list of all the costs computed during the optimization, this will 
            be used to plot the learning curve
    """

    costs = []
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layer_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = forward_propagation_deep(X, parameters, activation)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = backward_propagation_deep(AL, Y, caches, activation)
 
        # Update parameters
        parameters = update_parameters_deep(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        
        # Record the cost value every 100 iterations
        if i % 100 == 0:
            costs.append(cost)

    # plot the cost    
    if plot_lrn_curve == True:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters, costs

# In[8]: prediction
    
def predict_deep(X, Y, parameters, activation, print_accuracy = True):
    """
    This function is used to predict the results of an L-layer deep neural net
    
    Arguments:
    X -- data, numpy array of shape (dimension, number of examples)
    Y -- "true" binary labels vector (i.e. 0 if cat, 1 if non-cat), of shape: 
        (1, number of examples)
    parameters -- parameters of the trained model
    activation -- the activation function that was used to train the model in 
                    its hidden layers
    print_accuracy -- if True, it prints the value of accuracy
    
    Returns:
    predictions -- predictions for the given dataset X
    """
    
    m = X.shape[1]
#    n = len(parameters) // 2 # number of layers in the neural network
    
    predictions = np.zeros((1, m))
    
    # Forward propagation
    probabilities, caches = forward_propagation_deep(X, parameters, activation)

    # convert the probabilities to 0/1 predictions
    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0
    
    #print results
    if print_accuracy == True:
        print("Accuracy on this dataset is: "  + str(np.sum((predictions == Y)/m)*100) + "%")
        
    return predictions

# In[]:

