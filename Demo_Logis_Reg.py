# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:21:50 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>
@website: https://github.com/EsiSeraj/

Demo Script for "Logis_Reg.py", a logistic regression model generator with all
required helper functions

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
# # "Logis_Reg.py" test: Logistic Regression
# This module includes all of the helper functions for logistic regression
# included functions: "sigmoid()", "initialize_params()", propagate(), 
# optimize(), predict()
# =============================================================================

import Logis_Reg as LR

## sigmoid function
activation_val = LR.sigmoid(np.array([0, 2]))
print ("sigmoid([0, 2]) = " + str(activation_val))

## initialize parameters
dim = 2
case1, case2 = "zero", "random"
w1, b1 = LR.initialize_params(dim, case1)
w2, b2 = LR.initialize_params(dim, case2)
print("in case of " + str(case1) + " initialization, w=" + str(w1))
print("in case of " + str(case2) + " initialization, w=" + str(w2))

## forward and backward propagation
X, Y = np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
grads, cost = LR.propagate(w2, b2, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

## optimization
num_iterations, learning_rate = 1000, 0.009
params, grads, costs = LR.optimize(w2, b2, X, Y, num_iterations, learning_rate, print_cost = True)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

## prediction
labels = LR.predict(w2, b2, X)
print("predicted labels are y_hat=" + str(labels))

## logistic regression model
np.random.seed(1)
X_train = np.random.randn(100, 500)
Y_train = np.random.randint(2, size=(1, X_train.shape[1]))
X_test = np.random.randn(100, 200)
Y_test = np.random.randint(2, size=(1, X_test.shape[1]))
lr_mdl = LR.logis_reg_model(X_train, Y_train, X_test, Y_test, init_case = "zero", 
                    num_iter = 2000, learning_rate = 0.005, print_cost = True)

###############################################################################
###############################################################################
