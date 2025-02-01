import numpy as np
import math

# np element wise (will broadcast to fit etc.), just perfom the operation as normal math
# to do actual linear algebra operations, use np.dot, np.matmul, np.linalg.norm, etc.
# in a matrix (or any tensor) daxis = 0, will reduce the rows to a single row by summing up each column for example
# in a matrix (or any tensor) axis = 1, will reduce the columns to a single column by summing up each row for example

# only real numbers, we need to pass in tensors
def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s
# numpy can handle tensors -> np.exp(-x): [exp(-x), exp(-x), exp(-x), ...], if x is a tensor
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

# sigmoid derirvate = s(1-s)
def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

#  reshaping image to vector (length, width, 3 (rgb)) -> (length*width*3, 1) -> 1 column vector
def image2vector(image):
    shape = image.shape; # (length, width, 3)
    v = image.reshape(shape[0]*shape[1]*shape[2], 1)
    return v

# normalizing rows of a matrix
# change each row to have unit length
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    x = x/x_norm
    return x

# softmax function
# each eleemnt x, exp(x)/sum(exp(x) of all x in that row)
def softmax(x): 
    x_exp = np.exp(x)
    x_sum = np.sum(x, axis= 1, keepDim = True)
    s = x_exp/x_sum
    return s

# L1 loss function: L1(yhat, y) = sum(|y_i-yhat_i|) over all m examples
def L1(yhat, y):
    loss = np.sum(abs(y - yhat))
    return loss

# L2 loss function is the square of the difference between the predicted and actual values
def L2(yhat, y):
    loss = np.dot((y-yhat), (y-yhat))
    return loss

# since if y = [y1, y2, y3, ...], np.dot(y,y) = y1^2 + y2^2 + y3^2 + ..




