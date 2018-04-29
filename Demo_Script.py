# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:21:50 2018

@author: Esmaeil Seraj <esmaeil.seraj09@gmail.com>

Demo Script for Codes in This Repository
"""

import numpy as np

# =============================================================================
# # "Logis_Reg.py" test: Logistic Regression
# This module includes all of the helper functions for logistic regression
# included functions: "sigmoid()", "initialize_params()", propagate(), 
# optimize(), 
# =============================================================================

import Logis_Reg as LR
# "from Logis_Reg import sigmoid" this is another way of doing it, but it just
# imports the function that you just called. This way writting simply "sigmoid"
# is the command to use the sigmoid function instead of "LR.sigmoid()".

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
X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
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

## 

