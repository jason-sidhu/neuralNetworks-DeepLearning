import numpy as np

# basic sigmoid fxn
def sigmoid(x):
    return 1/(1+np.exp(-x))

# basic sigmoid derivative fxn
def sigmoid_derivative(x):
    # d/dxsig(x) = sig(x)*(1-sig(x))
    s = sigmoid(x)
    return s*(1-s)

# reshaping an image (length,width,depth) into a vector
# usually depth = 3 (RGB), and length and width are pixel dimensions
def image2vector(image):
    # turn into a column vector (length*width*depth,1) 
    shape = image.shape
    return image.reshape(shape[0]*shape[1]*shape[2],1)

# normalizing rows of a matrix
# sqrt(sum of squares of each element in a row)
#divide each element in a row by the row's norm
def normalizeRows(x):
    # compute the norm of each row, use the axis=1 to sum across the columns in a single row
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True) 
    # divide each row by its norm
    return x/x_norm

# softmax fxn
# softmax(x) = e^xij/sum(in a row e^xi)
def softmax(x): 
    x_soft = np.exp(x)
    x_sum = np.sum(x_soft, axis=1, keepdims=True)
    return x_soft/x_sum



