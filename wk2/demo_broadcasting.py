import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])

print(A)


cal = A.sum(axis=0) #sum vertically 
print(cal)
# [ 59.  239.  155.4  76.9]

#dividing the 3x4 matrix A by the 1x4 matrix cal through broadcasting
percentage = 100*A/cal.reshape(1,4) #broadcasting (cal is already 1,4)
print(percentage)

# [[94.91525424  0.          2.83140283 88.42652796]
#  [ 2.03389831 43.51464435 33.46203346 10.40312094]
#  [ 3.05084746 56.48535565 63.70656371  1.17035111]]