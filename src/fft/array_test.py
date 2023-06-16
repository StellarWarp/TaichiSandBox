import numpy as np
N = 16
M = 4

L = int(np.log(N)/np.log(M))

for d in range(L-1, -1, -1):
    log2_b = int(np.log2(M) * d)
    b = 1 << log2_b
    print(f'd = {d}, b = {b}, log2_b = {log2_b}')
    for k in range(N):
        w0 = k - k % b
        w = np.array([(k//b*b)*m for m in range(M)]) % N
        # w = np.array([(k - k % b)*m for m in range(M)])
        src_idx = np.array([(M*k - (M-1)*(k % b) + m*b) % N for m in range(M)])
        src_idx_ = np.array([(M*w0 - w0 + k + m*b) % N for m in range(M)])

        # w = k >> log2_b << log2_b
        print(f'k = {k}, src_idx = {src_idx}')
    print('------------------')


# print('graph LR')
# for l in range(0, L+1):
#     print(f'subgraph l_{l}[{l}]')
#     for k in range(N):
#         print(f'l_{l}_X_{k}[X_{k}]')
#     print('end')

# for l in range(1, L+1):
#     log2_b = int(np.log2(M) * (L - l))
#     b = 1 << log2_b
#     for k in range(N):
#         w0 = k - k % b
#         w = np.array([(k//b*b)*m for m in range(M)]) % N
#         # w = np.array([(k - k % b)*m for m in range(M)])
#         # src_idx = np.array([(M*k - k % b + m*b) % N for m in range(M)])
#         src_idx = np.array([(w0 - k + M*k + m*b) % N for m in range(M)])
#         for idx in src_idx:
#             print(f'l_{l-1}_X_{idx} ----> l_{l}_X_{k}')
#         # w = k >> log2_b << log2_b
#         # print(f'k = {k}, w = {w}, src_idx = {src_idx}')
#     print('')