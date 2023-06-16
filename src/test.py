import taichi as ti
import numpy as np
import math

ti.init(arch=ti.vulkan,kernel_profiler=True)

@ti.func
def q_log2(n:ti.u32):
    res = 0
    if n&ti.u32(0xffff0000):
        res += 16
        n >>= 16  
    if n&ti.u32(0x0000ff00):
        res += 8
        n >>= 8  
    if n&ti.u32(0x000000f0):
        res += 4
        n >>= 4  
    if n&ti.u32(0x0000000c):
        res += 2
        n >>= 2  
    if n&ti.u32(0x00000002):
        res += 1
        n >>= 1  
    return res

n = 1<<24
# debug_order = ti.field(ti.i32, shape=(int(math.log2(n)),n//2))
# order = ti.field(ti.i32, shape=1)

# @ti.kernel
# def prefix_sum_block_trans(m:ti.u32,offset:ti.i32,y:ti.template()):
#     block_sum:ti.u32 = 0
#     if offset == 0:
#         block_sum = 0
#     else:
#         block_sum = y[offset-1]
#     for i in range(m):
#         y[offset+i] += block_sum

@ti.kernel
def prefix_sum_interal(t:ti.u32,h:ti.i32,x:ti.template(),y:ti.template()):
    for i in range(h):
        src = ((i>>t)<<(t+1)) + (1<<t) - 1
        dst = src + 1 + (i & ((1<<t) - 1))
        y[dst] += y[src]
    

def prefix_sum(x:ti.template(),y:ti.template()):
    n = x.shape[0]
    y.copy_from(x)
    total_step = int(np.log2(n))
    h = n // 2
    for t in range(total_step):
        prefix_sum_interal(t,h,x,y)
# def prefix_sum(x:ti.template(),y:ti.template()):
#     n = x.shape[0]
#     y.copy_from(x)
#     m = n // 4
#     block_step = n//m
#     total_step = int(np.log2(m))
#     h = m // 2
#     offset = 0
#     for _ in range(block_step):
#         for t in range(total_step):
#             prefix_sum_interal(t,h,offset,x,y)
#             prefix_sum_block_trans(m,offset,y)
#         offset += m


@ti.kernel
def zero_count(x:ti.template(),bit_i:ti.i32,zero:ti.template()):
    for i in range(x.shape[0]):
        if x[i] & (1<<bit_i):
            zero[i] = ti.i8(0)
        else:
            zero[i] = ti.i8(1)

# debug
zero_map = ti.field(ti.i32,shape=n)
one_map = ti.field(ti.i32,shape=n)
map = ti.field(ti.i32,shape=n)


@ti.kernel
def get_map(
    zero:ti.template(),
    zero_sum:ti.template(),
    x: ti.template(),
    y: ti.template()):
    total_zero = zero_sum[zero_sum.shape[0]-1]
    for i in range(zero_sum.shape[0]):
        zero_map_i = zero_sum[i-1] if i > 0 else 0
        one_map_i = i + total_zero - zero_map_i
        map_i = zero_map_i if zero[i] == 1 else one_map_i
        y[map_i] = x[i]

        # # debug
        # zero_map[i] = zero_map_i
        # one_map[i] = one_map_i
        # map[i] = map_i



def radix_sort(x:ti.template()):
    # dtype of x is ti.u32
    len = x.shape[0]
    zero = ti.field(ti.i8,shape=len)
    zero_sum = ti.field(ti.i32,shape=len)
    y = ti.field(ti.u32,shape=len)
    for bit_i in range(32):
        (src,dst) = (y,x) if bit_i & 1 else (x,y)
        zero_count(src,bit_i,zero)
        prefix_sum(zero,zero_sum)
        get_map(zero,zero_sum,src,dst)

        # # debug
        # print(f'\nbit_i = {bit_i}\n')
        # print(f'zero    {zero.to_numpy()}')
        # print(f'zero_sum{zero_sum.to_numpy()}')
        # print(f'zero_map{zero_map.to_numpy()}')
        # print(f'one_map {one_map.to_numpy()}')
        # print(f'map     {map.to_numpy()}')
        # dst_ = dst.to_numpy()
        # for i in range(len):
        #     print(f'{dst_[i]:06b}')
        
import time
print('test')
print(f'generate data of {n}')
#random
cpu_data = np.random.randint(0,2**16,n,dtype=np.uint32)

gpu_data = ti.field(ti.u32,shape=n)
gpu_data.from_numpy(cpu_data)


gpu_time = 0
cpu_time = 0

print('start sort')
gpu_time = time.time()
ti.profiler.clear_kernel_profiler_info()
radix_sort(gpu_data)
ti.profiler.print_kernel_profiler_info()
gpu_time = time.time() - gpu_time
print(f'gpu_time = {gpu_time}')

cpu_time = time.time()
cpu_data.sort()
cpu_time = time.time() - cpu_time
print(f'cpu_time = {cpu_time}')


# diff = 0
# gpu_data_ = gpu_data.to_numpy()
# diff = gpu_data_ - cpu_data
# print(f'error = {max(diff)}')





# def ref(x):
#     res = np.array([0 for i in range(len(x))])
#     for i in range(len(x)):
#         res[i] = res[i-1] + x[i]
#     return res

# input = ti.field(ti.u32, shape=n)
# output = ti.field(ti.u32, shape=n)
# for i in range(n):
#     input[i] = i
# prefix_sum(input,output)
# gpu_output = output.to_numpy()
# cpu_output = ref(input.to_numpy())
# diff_count = 0
# diff_begin = -1
# for i in range(n):
#     if gpu_output[i] != cpu_output[i]:
#         diff_count += 1
#         if diff_begin == -1:
#             diff_begin = i
# print()
# print(f'error count = {diff_count}')
# print(f'error since {diff_begin}')
# debug_res = debug_order.to_numpy()
# for i in range(debug_res.shape[0]):
#     # sort
#     debug_res[i,:] = np.sort(debug_res[i,:])
#     for j in range(debug_res.shape[1]):
#         print(f'{debug_res[i,j]:3d}',end=' ')
#     print()


# print('max in step')
# for i in range(debug_res.shape[0]):
#     print(f'{debug_res[i,:].max():3d}',end=' ')
# print()
# print('min in step')
# for i in range(debug_res.shape[0]):
#     print(f'{debug_res[i,:].min():3d}',end=' ')
# print()

# 统计每一步的出现的数字的个数,显示数字与其对应的个数