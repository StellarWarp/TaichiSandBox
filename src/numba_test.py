import numba
import numpy as np
from numba import njit
from numba import cuda

@njit
def f(a):
    sum = 0
    for i in range(a.shape[0]):
        sum += a[i, i]
    return sum

#parallel test
@njit(parallel=True)
def g(a):
    sum = 0
    for i in numba.prange(a.shape[0]):
        sum += a[i, i]
    return sum

#cude test
@cuda.jit
def prefix_sum_interal(total_step,x):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    for t in range(total_step):
        src = ((i>>t)<<(t+1)) + (1<<t) - 1
        dst = src + 1 + (i & ((1<<t) - 1))
        x[dst] += x[src]
        # sync all threads in every block
        cuda.syncthreads()


    
def prefix_sum(x, y):
    n = x.shape[0]
    for i in range(n):
        y[i] = x[i]
    total_step = int(np.log2(n))
    h = n // 2

    prefix_sum_interal[h//32,32](total_step,y)


x = np.random.randint(0,10,512)
x_d = cuda.to_device(x)
y_d = cuda.device_array_like(x_d)
prefix_sum(x_d,y_d)
def test(x):
    sum = 0
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        sum += x[i]
        y[i] = sum
    return y
print(y_d.copy_to_host())
print(test(x))


   

