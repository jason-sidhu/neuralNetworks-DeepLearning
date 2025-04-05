# Implement the functions required to build a deep neural network
# Will use these functions to then build a deep neural network for image classification

import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import copy


#----INIT-----
# Initialize Parameters for a 2-layer neural network
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments: 
    n_x: size of the input layer
    n_h: size of the hidden layer
    n_y: size of the output layer
        
    
    Returns dictionary parameters
    W1: weight matrix (n_h, n_x)
    b1: bias vector of shape (n_h, 1)
    W2: weight matrix of (n_y, n_h)
    b2: bias vector of shape (n_y, 1)
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2
    }
    return parameters

# Initialize Parameters for a L-layer neural network
def initialize_parameters_deep(layer_dims):
    """
    Arguments: 
    layer_dims: python list that contains the dimension of each layer in our network

    return dictionary parameters containing parameteres W1, b1, up till Wl and bl
    """
    # We know that in order to correct compute activations of layer l, based on layer l - 1
    # Wl is of shape (n_l, n_l-1) -> where n_l = layer_dims, n_l-1 = layer_dims - 1
    # bl is of shape (n_l, 1) -> bc it's a vector bias

    parameters = {} #empty dict
    L = len(layer_dims) #number of layers (including input)
    # W1 to W(l-1) -> since the first layer is the input, we start at 1.
    # And L is also accounting for that so we go up to but not including l
    # The Wl-1 and bl-1 connect the l-1th layer to the lth layer
    for i in range(1, L): 
        # random.randn does a stanard normal distribution, mean = 0, std = 1
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*0.01 #prevent being to large to start.
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))


    return parameters

#-----FORWARD PROPAGATION MODULE-----
# Linear -> Z[l] = W[l]A[l-1] + b[l] (vectorized over all examples)
# After this we apply the activation (ReLU or Sigmoid)
# Our enitre model will be [Linear -> ReLU activation] * (L-1 times) -> Linear -> Sigmoid activation for final prediction
# Linear Forward
def linear_forward(A, W, b):
    """
    Linear part of a layer's forward prop

    Arguments: A - activations from prev layer (or inputs) -> (size of prev layer, num of training examples)
    W: Weights matrix for theis layer, should be numpy array of shape (size of current layer, size of pre layer)
    b: bias vector: numpy array of shape (size of curr layer, 1)

    returns
    Z - input to the activation function having applied the linear transformation (pre-activation)
    cache - python tuple with "A", "W", "b": stored for computing the backward pass effictiently
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the Linear->Activation layer
    
    Arguments:
    A_prev: activations from prev layer (or inputs) -> (size of prev layer, num of training examples)
    W: Weights matrix for theis layer, should be numpy array of shape (size of current layer, size of pre layer)
    b: bias vector: numpy array of shape (size of curr layer, 1)

    returns :
    A: the output of the activation func, post-activation, size: (size of curr layer, num examples)
    cache - python tuple containing "linear_cache" and "activation_cache"
            Stored for computing backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid": 
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache
 
# Implement the entire L_model_forward block 
def L_model_forward(X, parameters): 
    """
    Implement the forward prop, (L-1)*linear->relu -> linear->sigmoid computation in one func

    Arguments:
    X -> data, numpy array of size (input size [num neurons in input], training examples)
    parameters -> output from init_parameters_deep() -> the weights and biases initialized

    returns: 
    AL: activation value from the last layer (the prediction)
    caches - list of cahces containing: 
            every cach of linear_activation_forward, L of them from 0 to L-1
            They have linear caches (A, W, b) and activation cache (Z) for each layer
    """

    caches = [] #list of caches
    A = X #init A[0] to X to start
    L = len(parameters)//2 #num of layers (since parameters as 2 parameters for each layer)

    # [LINEAR->RELU] * (L-1) times
    # start at 1 bc layer 0 is input
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    # LINEAR->SIGMOID
    AL, cache = linear_activation_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

#-----COST-----
def compute_cost(AL, Y):
    """
    Compute cost
    Arguments:
    AL: the probability vvector that has the final predicitons, shape(1, number of examples) -> since we have binary classification
    Y: true label vector (0 for non cat, 1 for cat)
    Returns:
    cost: cross-entropy cost
    """

    m = Y.shape[1]
    cost = (-1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    # to make sure cost shape is correct
    cost = np.squeeze(cost)

    return cost

#-----BACKWARD PROPAGATION MODULE-----
# [LINEAR -> RELU]  ×  (L-1) -> LINEAR -> SIGMOID backward (whole model)
def linear_backward(dZ, cache):
    """
    linear portion of backward prop for single layer

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*(np.dot(dZ, A_prev.T))
    db = (1/m)*(np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    dA: post activation gradient for curr layer l
    cache: tuple of values (linear_cache, activation_cache), stored from forward layer
    activation: activation used in the layer (sigmoid, relu)

    returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    Order of the prop is actually sigmoid->linear->(relu->linear)*L-1
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) #num of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    current_cache = caches[L-1]
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # loop from l = L-2 to l = 0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp


    return grads


#---UPDATE PARAMETERS-----
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate*grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate*grads["db" + str(l+1)])
        
    return parameters

def main():
    # TESTS OF HELPER FUNCTIONS
    # parameters = initialize_parameters(2, 2, 1)
    # print(parameters)

    # parameters = initialize_parameters_deep([5,4,3])
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

if __name__ == "__main__":
    main()