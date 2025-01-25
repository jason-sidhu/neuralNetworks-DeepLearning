import numpy as np
import time

a = np.array([1,2,3,4])
print(a)

# random 1-d tensor (array) of 1,000,000 elements
a = np.random.random(1_000_000)
b = np.random.random(1_000_000)
print(a)
print(b)

tic = time.time()
c = np.dot(a, b) 
toc = time.time()

# ~0.39ms 
print(f"c = {c} vectorized verison: {str(1000*(toc-tic))} ms")

c = 0
tic = time.time()
for i in range(1_000_000):
    c += a[i]*b[i]

toc = time.time()
#~365ms
print(f"c = {c} loop verison: {str(1000*(toc-tic))} ms")

# vectorization much faster -> allows for parrellelization of hardware through SIMD (single instruction, multiple data) instructions
