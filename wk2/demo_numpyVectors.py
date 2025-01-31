import numpy as np


a = np.random.randn(5) #random 1-d tensor (array) of 5 elements
print(a)
print(a.shape) # (5,)

print(a.T) # [ 0.  0.  0.  0.  0.] -> no change

print(np.dot(a, a.T)) # 5.0 -> dot product of a and a.T -> scalar (not as expected)

a = np.random.randn(5,1) #random 2-d tensor (array) of 5x1 elements
print(a)
print(a.shape) # (5,1) [5 rows, 1 column] [[0], [0], [0], [0], [0]]
print(a.T) # [[ 0.  0.  0.  0.  0.]] -> transpose of a -> 1x5
print(np.dot(a, a.T)) # [[ 0.  0.  0.  0.  0.]
                      #  [ 0.  0.  0.  0.  0.]
                      #  [ 0.  0.  0.  0.  0.]
                      #  [ 0.  0.  0.  0.  0.]
                      #  [ 0.  0.  0.  0.  0.]] -> 5x5 matrix

assert(a.shape == (5,1)) #assertion error if not true

 