# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:21:50 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>
@website: https://github.com/EsiSeraj/

Demo Script for "Plain_DNN.py", a plain Deep Neural Network (Without any 
improvement technique, a plain implementation of deep neural networks) model
generator with all required helper functions

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

import numpy as np

# =============================================================================
# "Plain_DNN.py" test: plain Deep Neural Network (Without any improvement tech)
# This module includes all of the helper functions for plain deep Neural Net
# included functions: sigmoid(), relu(), tanh(), sigmoid_backward(), 
# relu_backward(), tanh_backward(), initialize_parameters_deep(), 
# linear_forward(), activation_forward(), forward_propagation_deep(), 
# compute_cost(), linear_backward(), activation_backward(), predict_deep(), 
# backward_propagation_deep(), update_parameters_deep(), plain_nn_model_deep(),
# =============================================================================

import Plain_DNN as pDNN

## initialize parameters
layers_dims = [10, 4, 6, 1]
parameters = pDNN.initialize_parameters_deep(layers_dims)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("W3 = " + str(parameters["W3"]))
print("b3 = " + str(parameters["b3"]))

## forward propagation
np.random.seed(1)
X_train = np.random.randn(layers_dims[0], 500)
Y_train = np.random.randint(2, size = (1, X_train.shape[1]))
X_test = np.random.randn(layers_dims[0], 200)
Y_test = np.random.randint(2, size = (1, X_test.shape[1]))
activation = 'relu'
y_hat, caches = pDNN.forward_propagation_deep(X_train, parameters, activation)
print("y_hat = " + str(y_hat))
print("Length of caches list = " + str(len(caches)))

## compute the cost function
c = pDNN.compute_cost(y_hat, Y_train)
print("cost = " + str(c))

## backward propagation
gradients = pDNN.backward_propagation_deep(y_hat, Y_train, caches, activation)
print("\n check the gradients dictionary in Variable Explorer window..!!\n")

## update parameters
learning_rate = 0.001
params1 = pDNN.update_parameters_deep(parameters, gradients, learning_rate)
print("\n check the updated parameters dictionary in Variable Explorer window..!!\n")

## plain deep neural network model generator
params2, costs = pDNN.plain_nn_model_deep(X_train, Y_train, layers_dims, activation, learning_rate, 
                        num_iterations = 3000, print_cost = True, plot_lrn_curve = True)

## prediction using the output of a trained deep model
preds = pDNN.predict_deep(X_train, Y_train, parameters, activation, print_accuracy = True)

###############################################################################
###############################################################################
