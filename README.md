# Basic_Plain_DNN
This repository includes the very basic Deep Learning modules (Logistic Regression - Plain Shallow Networks - Plain Deep Networks) in Python, for starters. These implementations do not include any of the improvement techniques (i.e. regularization, batch normalization, drop-out and etc) for deep neural network model. Basically, they just implement the general (and complete) architecture of a deep model training problem in a vectorized and very efficientand approach through providing all of the required helper functions first. Shallow neural networks with only one hidden layer and logistic regression models are also included in current repository. The main steps for building a Neural Network that are implemented here are:

     1. Define the model structure (i.e. number and dimensions of different layers, activation functions and etc) 
     2. Initialize the model's parameters and set the hyperparameters
     3. Loop:
         - Calculate current loss (forward propagation)
         - Calculate current gradient (backward propagation)
         - Update parameters (gradient descent)
     4. Use learnt parameters to predict the labels on train data (use forward path)
     5. Use learnt parameters to predict the labels on test data

# Overview
 Features of current repository can be summrized as below:
 
	1. Plain architectures implemented from scratch (great for beginners)
	2. Everything in raw Python codes and not using TensorFlow, Torch, PyTorch or etc (great for beginners to understand the basic concepts and learn how DNN works)
	3. Vectorized implementations (very efficient)
	4. Numpy implementation for all computations (very efficient)
	5. Providing all of the helper functions in a module (you can build upon this plain model and add your own helper functions)
	6. Separate implementations for Deep NN, Shallow NN and Logistic Regression (great for beginners to interact with each separately)
	7. Demo script provided for each main module (good for starters)
	8. Only package dependency: "numpy" (also "matplotlib" in case you want to plot anything!)

# User Manual
List of the functions and scripts are as below with a short description of each. For each method, use the provided Demo script to see how the module works and what type of helper functions are provided within that module.

	1. **"Plain_DNN.py":** Plain Deep Neural Network (Without any improvement technique, a plain, vectorized, highly efficient, all raw Python implementation of deep neural networks)
		- All required helper functions: included functions: sigmoid(), relu(), tanh(), sigmoid_backward(), relu_backward(), tanh_backward(), initialize_parameters_deep(), linear_forward(), activation_forward(), forward_propagation_deep(), compute_cost(), linear_backward(), activation_backward(), predict_deep(), backward_propagation_deep(), update_parameters_deep(), plain_nn_model_deep()
		- tanh, relu and sigmoid activations for hidden layers
		- Sigmoid output for binary classification
		- A comprehensive but plain DNN model generator
	
	2. **"Demo_Plain_DNN.py":** Demo Script to test the hellper functions in "Plain_DNN.py"
	
	3. **"Shallow_NN.py":** Shalllow Neural Network (one hidden layer) - SNN
		- All required helper functions: included functions: layer_sizes(), sigmoid(), initialize_params(), forward_propagation(), cross_ent_cost(), backward_propagation(), update_parameters(), shallow_nn_model(), predict()
		- tanh activation for hidden layer
		- Sigmoid output for binary classification
		- A comprehensive SNN model generator
	
	4. **"Demo_Shallow_NN.py":** Demo Script to test the hellper functions in "Shallow_NN.py"
	
	5. **"Logis_Reg.py":** Logistic Regression Model Generator and Predictor
		- All required helper functions: included functions: "sigmoid()", "initialize_params()", propagate(), optimize(), predict()
		- Sigmoid output for binary classification
		- A comprehensive logistic regression model generator
		
	6. **"Demo_Logis_Reg.py":** Demo Script to test the hellper functions in "Logis_Reg.py"
	
# Update Repository
This repository is pretty new and so gets regular updats; I also intend to create another repository in my GitHub with more professional (with improvement techniques) but simple implementations of deep learning methods, which is gonna be great for beginners in this area. 

Also any comments on the functions (improvement recommendations) and suggestions (on what other helper functions should be added) are greatly welcomed and appreciated. Feel free to contact me here or directly by email.

# License - No Warranty

Copyright (C) <2018> GNU General Public License
    
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
