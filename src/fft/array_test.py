import numpy as np
# N = 16
# M = 4

# L = int(np.log(N)/np.log(M))

# for d in range(L-1, -1, -1):
#     log2_b = int(np.log2(M) * d)
#     b = 1 << log2_b
#     print(f'd = {d}, b = {b}, log2_b = {log2_b}')
#     for k in range(N):
#         w0 = k - k % b
#         w = np.array([(k//b*b)*m for m in range(M)]) % N
#         # w = np.array([(k - k % b)*m for m in range(M)])
#         src_idx = np.array([(M*k - (M-1)*(k % b) + m*b) % N for m in range(M)])
#         src_idx_ = np.array([(M*w0 - w0 + k + m*b) % N for m in range(M)])

#         # w = k >> log2_b << log2_b
#         print(f'k = {k}, src_idx = {src_idx}')
#     print('------------------')



import taichi as ti

ti.init(arch=ti.gpu)


def static_list():
    return [i for i in range(32)],32

@ti.kernel
def alloc_array(size:ti.template()):

    vec = ti.Vector([[x for x in range(4)]for i in range(size)])

    print(vec.get_shape())
    for i in range(size):
        for j in range(4):
            print(vec[i,j])

    for i in range(size):
        for j in range(4):
            vec[i,j] += 1
    print('------------------')
    for i in range(size):
        for j in range(4):
            print(vec[i,j])

        
    


alloc_array(4)