import numpy as np
import os
import copy
import matplotlib.pyplot as plt #to plot graphs in python
import h5py #common package to interact with a dataset that is stored on an H5 file
import scipy
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from Utils import load_dataset


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# setup weights and bias, dim is the number of features(n = num_px * num_px * 3)
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1)) #column vector
    b = 0
    return w, b 

"""
w - weights: a numpy array of size (num_px * num_px * 3, 1) (n x 1)
b - bias: a scalar
X - data of size (num_px * num_px * 3, number of examples) (n x m)
Y - true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
return the computed gradients, dw and db, to perform one step in the gradient descent algorithm
"""
def propagate(w, b, X, Y):
    m = X.shape[1]
    YHAT = sigmoid(np.dot(w.T, X) + b) #prediction yhat = sigmoid(w.T * X + b), to turn w into (1xn) x is (nxm) = (1xm) row vector
    cost = (-1/m)*np.sum(Y*np.log(YHAT) + (1-Y)*np.log(1-YHAT)) #cost function: J = (-1/m) * sum(y*log(yhat) + (1-y)*log(1-yhat))
    dw = (1/m)*np.dot(X, (YHAT - Y).T) # YHAT - Y: (1xm), x is (nxm), so we are doing (nxm)(mx1) = (nx1) col vector for each change in w(i) 1->n
    db = (1/m)*np.sum(YHAT - Y) # (1xm) row vector, so summing it will give a scalar -> reducesum
    cost = np.squeeze(np.array(cost)) #squeeze to remove extra dimensions
    grads = {"dw": dw, "db": db} #dictionary to store gradients
    return grads, cost

"""
run the gradient descent algorithm to learn w and b by minimizing the cost function J
return params with w and b. 
"""
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost = False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    for i in range(num_iterations):
        # one step of gradient descent
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate*db
        # only record cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

def predict(w, b, X):
    # given w, b, x, we can predict for an entire training set. or just one example
    m = X.shape[1]
    Y_predictions = np.zeros((1, m)) #row vector
    YHAT = sigmoid(np.dot(w.T, X) + b) #prediction yhat = sigmoid(w.T * X + b), to turn w into (1xn) x is (nxm) = (1xm) row vector
    for i in range(YHAT.shape[1]): #iterate over all tests
        if YHAT[0, i] > 0.5:
            Y_predictions[0, i] = 1
        else:  
            Y_predictions[0, i] = 0
    return Y_predictions #with all predictions

# full model to train, and then test. 
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    #initialize weights and bias, pass in dimensions
    w, b = initialize_with_zeros(X_train.shape[0]) 

    #optimize the weights and bias
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    #get the weights and bias
    w = params["w"]
    b = params["b"]

    #predict the training and test set
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def main():
    # Load dataset
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    print(f"Train set shape: {train_set_x_orig.shape}, Labels: {train_set_y.shape}")
    print(f"Test set shape: {test_set_x_orig.shape}, Labels: {test_set_y.shape}")
    
    # verify a picture from the dataset
    # Example of a picture
    index = 8
    # Display the image 
    # plt.imshow(train_set_x_orig[index])  # train_x is already in (64,64,3) format
    # plt.axis("off")  # Hide axis for better visualization
    # plt.title(f"Label: {classes[int(train_set_y[0, index])]}")  # Get the label and decode it
    # plt.show() #UNCOMMENT TO SEE IMAGE, CLOSE THE IMAGE TO CONTINUE EXECUTION, BLOCKING WINDOW

    # verify num of training examples, num of test examples, num_px (= height = width of a training image)
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_set_x_orig.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x shape: " + str(test_set_x_orig.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    # Reshape the training and test examples -> To flatten into a column vector
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print ("train_set_y shape: " + str(train_set_y.shape))
    print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print ("test_set_y shape: " + str(test_set_y.shape))

    # standardize dataset (subtract by mean and divide by std dev, but here for pixels just divide by 255)
    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255

    logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

    # Plot learning curve (with costs)
    costs = np.squeeze(logistic_regression_model['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    # plt.show()

    # Test with own image afterwards and change this to the name of your image file and using our parameters
    my_image = "nik2.jpg"       
    fname = "images/" + my_image
    image = np.array(Image.open(fname).resize((num_px, num_px)))
    image = image / 255.
    image = image.reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")





if __name__ == "__main__":
    main()