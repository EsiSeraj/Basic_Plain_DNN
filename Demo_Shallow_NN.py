# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:21:50 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>
@website: https://github.com/EsiSeraj/

Demo Script for "Shallow_NN.py", a Shalllow Neural Network (one hidden layer) -
 (SNN) model generator with all required helper functions

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
# "Shallow_NN.py" test: Shalllow Neural Network (one hidden layer)
# This module includes all of the helper functions for Shalllow Neural Network
# included functions: layer_sizes(), sigmoid(), initialize_params(),
# forward_propagation(), cross_ent_cost(), backward_propagation(), 
# update_parameters(), shallow_nn_model(), predict() 
# =============================================================================

import Shallow_NN as SNN

## parameter initialization
n_h = 4
np.random.seed(1)
X_train = np.random.randn(100, 500)
Y_train = np.random.randint(2, size = (1, X_train.shape[1]))
X_test = np.random.randn(100, 200)
Y_test = np.random.randint(2, size = (1, X_test.shape[1]))

n_x, n_h, n_y = SNN.layer_sizes(X_train, Y_train, n_h)
params = SNN.initialize_params(n_x, n_h, n_y)
print("W1 = " + str(params["W1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["W2"]))
print("b2 = " + str(params["b2"]))

## forward propagation
y_hat, cache = SNN.forward_propagation(X_train, params)
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), 
      np.mean(cache['A2']))

## cost function
c = SNN.cross_ent_cost(y_hat, Y_train)
print("cost = " + str(c))

## back propagation
gradients = SNN.backward_propagation(params, cache, X_train, Y_train)
print("dW1 = " + str(gradients["dW1"]))
print("db1 = " + str(gradients["db1"]))
print("dW2 = " + str(gradients["dW2"]))
print("db2 = " + str(gradients["db2"]))

## updating model parameters
learning_rate = 1.2
params1 = SNN.update_parameters(params, gradients, learning_rate)
print("W1 = " + str(params1["W1"]))
print("b1 = " + str(params1["b1"]))
print("W2 = " + str(params1["W2"]))
print("b2 = " + str(params1["b2"]))

## buil SNN model
num_iterations = 2000
learning_rate = 0.005
plot_flag = True
print_cost = True
params2, costs = SNN.shallow_nn_model(X_train, Y_train, n_h, num_iterations, 
                                     learning_rate, print_cost, plot_flag)

## prediction using built snn model
predict_train = SNN.predict(params, X_train)
predict_test = SNN.predict(params, X_test)
print ('Train Accuracy: %d' % float((np.dot(Y_train, predict_train.T) + 
                                     np.dot(1-Y_train,1-predict_train.T))/float(Y_train.size)*100) + '%')
print ('Test Accuracy: %d' % float((np.dot(Y_test,predict_test.T) + 
                                    np.dot(1-Y_test,1-predict_test.T))/float(Y_test.size)*100) + '%')

###############################################################################
###############################################################################
