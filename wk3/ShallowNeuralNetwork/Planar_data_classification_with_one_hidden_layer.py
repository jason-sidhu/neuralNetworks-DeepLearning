#binary classification with single hidden layer neural network
import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01 #weight matrix (new layer neurons size, prev layer neurons size) (n_h, n_x)
    b1 = np.zeros((n_h, 1)) #bias vector (n_h, 1), which gets broadcasted to all m examples
    W2 = np.random.randn(n_y, n_h) * 0.01 #weight matrix (output layer size, new layer size) (n_y, n_h),  (1, n_h)
    b2 = np.zeros((n_y, 1)) #bias vector (n_y, 1), which gets broadcasted to all m examples, so just scalar right now
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    # get all the parameters from initialization (or prev iteration with update)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # forward Propagation to calculate A2 (which is yhat or the probability for outcome)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)    
    assert(A2.shape == (1, X.shape[1])) #predictions should be <1, m> 
    # return the forward propogation results to use in back propogation and cost calculation
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def compute_cost(A2, Y):
    # A2 is predictions <1, m>, Y is true labels <1, m>
    m = Y.shape[1] # number of examples
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = -(1/m)*np.sum(logprobs)        
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # example will turn [[17]] into 17 
    return cost

def update_parameters(parameters, grads, learning_rate = 1.2):
    # get a copy of each parameter from the dictionary that we need to update
    #avoid modifying the original dictionary
    W1 = copy.deepcopy(parameters["W1"])
    W2 = copy.deepcopy(parameters["W2"])
    b1 = parameters["b1"]
    b2 = parameters["b2"]    
    #get the gradients used to update the parameters
    # YOUR CODE STARTS HERE
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # gradient descent step for each parameter with the learning ra
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    # return updated parameters
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    #get parameters needed to calculate gradients
    W1 = parameters["W1"]
    W2 = parameters["W2"]    
    A1 = cache["A1"]
    A2 = cache["A2"]
    # backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims = True)
    # return the gradients needed to do a step of gradient descent
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def nn_model(X, Y, n_x, n_h, n_y, num_iterations = 10000, print_cost=False):
    # 2) initialize the model's parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    #3 in a loop
    for i in range(0, num_iterations):
        # a) forward propagation
        A2, cache = forward_propagation(X, parameters)
        # b) compute cost
        cost = compute_cost(A2, Y)
        # c) backward propagation
        grads = backward_propagation(parameters, cache, X, Y)
        # d)update parameters (gradient descent)
        parameters = update_parameters(parameters, grads)
          # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    # return final parameters to use for testing
    return parameters

def predict(parameters, X):
    # with the the parameters already trained, we can use them to predict the outcome with one forward pass on our data X
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5) # if the probability is greater than 0.5, we predict 1, else 0    
    return predictions



def main():
    # use utils to gather data set (datset comes from sklearn)
    X, Y = load_planar_dataset()
    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

    # get shape of data and understand training size
    shape_X = X.shape
    shape_Y = Y.shape
    # X = [ x(1){as a col vector}, x(2), ..., x(m)] there for shape_X[0] = feastures, and shape_X[1] = training set size = m
    m = shape_X[1] # training set size


    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print (f'n input features = {shape_X[0]}, m = {m} training examples!') 
    # n = 2, bc the input is a set of planar data, so x1 and x2 coordinates of the dots, y is the label of 0 or 1 (red or blue)
    
    # we will use non linear activation functions. Especially important if non linear data, but also for neural networks. If
    # we just use linear activation functions, the layers dont add anything and you have ultimately a linear model.
        # following the general methodology of setting up Neural Network
    # 1) define the nn structure. Set number of neurons for input layer, hidden layer, and output layer
    n_x = shape_X[0]
    n_h = 4 # chosen for this model
    n_y = shape_Y[0]
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h)) 
    print("The size of the output layer is: n_y = " + str(n_y)) 
    # rest of the steps in the nn_model function where we train the model on X and Y
    parameters = nn_model(X, Y, n_x, n_h, n_y, num_iterations = 10000, print_cost=True)
    # test the model on the training set
    # plot the decision boundary prediction
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

    # # extra datasets from sklearn to do testing on, uncomment to test and see decision boundary
    # noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    # datasets = {"noisy_circles": noisy_circles,
    #             "noisy_moons": noisy_moons,
    #             "blobs": blobs,
    #             "gaussian_quantiles": gaussian_quantiles}
    
    # # try different datasets
    # dataset = "noisy_moons" 
    # X, Y = datasets[dataset]
    # X, Y = X.T, Y.reshape(1, Y.shape[0])

    # # make blobs binary
    # if dataset == "blobs":
    #     Y = Y%2
    # # Visualize the data
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == "__main__":
    main() 
