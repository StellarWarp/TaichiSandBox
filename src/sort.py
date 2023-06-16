import taichi as ti
import numpy as np

@ti.func
def log2_int(n:ti.u32):
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


@ti.kernel
def prefix_sum_interal(t:ti.u32,h:ti.i32,y:ti.template()):
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
        prefix_sum_interal(t,h,y)

@ti.func
def mutex_sync(mutex:ti.i32,block_count:ti.i32,thread_id:ti.i32):
    if thread_id == 0:
        ti.atomic_add(mutex,1)
        print(f'mutex = {mutex}')
        while mutex != block_count:
            pass
        mutex = 0
    ti.simt.block.sync()

ti.init(arch=ti.gpu,kernel_profiler=True)


@ti.kernel
def prefix_sum_kernel(x:ti.template(),y:ti.template()):
    n = x.shape[0]
    for i in x:
        y[i] = x[i]
    total_step = log2_int(n)
    threads = n // 2
    # block_dim = 64
    block_count = threads // 64
    mutex = 0
    ti.loop_config(block_dim=64)
    for i in range(threads): #parallel
        thread_gid = ti.global_thread_idx()
        block_id = thread_gid // 64 
        thread_id = thread_gid % 64 # ti.simt.block.thread_idx() unavailable for cuda
        #loop
        for t in range(total_step):
            src = ((i>>t)<<(t+1)) + (1<<t) - 1
            dst = src + 1 + (i & ((1<<t) - 1))
            y[dst] += y[src]
            # sync all threads
            if thread_id == 0:
                ti.atomic_add(mutex,1)
                while mutex < block_count * (t+1):
                    print(f'mutex = {mutex} block_count = {block_count * (t+1)} block_id = {block_id}')
                    pass
                # print(f'passed mutex = {mutex} block_count = {block_count * (t+1)} block_id = {block_id}')
            ti.simt.block.sync()
            
            


@ti.kernel
def zero_count(x:ti.template(),bit_i:ti.i32,zero:ti.template()):
    for i in range(x.shape[0]):
        if x[i] & (1<<bit_i):
            zero[i] = ti.i8(0)
        else:
            zero[i] = ti.i8(1)


@ti.kernel
def get_map(
    zero:ti.template(),
    zero_sum:ti.template(),
    x: ti.template(),
    y: ti.template(),
    map: ti.template()):

    total_zero = zero_sum[zero_sum.shape[0]-1]
    for i in x:
        zero_map_i = zero_sum[i-1] if i > 0 else 0
        one_map_i = i + total_zero - zero_map_i
        map_i = zero_map_i if zero[i] == 1 else one_map_i
        y[map_i] = x[i]
        # x now store for map
        x[i] = map_i
    for i in map:
        map[i] = x[ti.i32(map[i])]

@ti.kernel
def map_table_init(map:ti.template()):
    for i in map:
        map[i] = i

def radix_sort(x:ti.template(),index_map:ti.template()):
    len = x.shape[0]

    zero = ti.field(ti.i8,shape=len)
    zero_sum = ti.field(ti.i32,shape=len)
    tb1 = ti.field(ti.u32,shape=len)
    tb2 = ti.field(ti.u32,shape=len)
    tb1.copy_from(x)
    src_dst = [(tb2,tb1)if i & 1 else (tb1,tb2) for i in range(32)]

    map_table_init(index_map)
    for bit_i in range(32):
        (src,dst) = src_dst[bit_i]
        # print(f'src =      {src.to_numpy()}')
        zero_count(src,bit_i,zero)
        prefix_sum(zero,zero_sum)
        get_map(zero,zero_sum,src,dst,index_map)
        # print(f'zero =     {zero.to_numpy()}')
        # print(f'zero_sum = {zero_sum.to_numpy()}')
        # print(f'map_i =    {src.to_numpy()}')
        # print(f'map =      {index_map.to_numpy()}')

# #test
# ti.init(arch=ti.gpu,kernel_profiler=True)
test_data = np.random.randint(0,1<<6,1<<10,dtype=np.uint32)
x = ti.field(ti.u32,shape=test_data.shape)
x.from_numpy(test_data)
y1 = ti.field(ti.u32,shape=test_data.shape)
y2 = ti.field(ti.u32,shape=test_data.shape)
ti.profiler.clear_kernel_profiler_info()
prefix_sum(x,y1)
ti.profiler.print_kernel_profiler_info()
ti.profiler.clear_kernel_profiler_info()
prefix_sum_kernel(x,y2)
ti.profiler.print_kernel_profiler_info()


y1_host = y1.to_numpy()
y2_host = y2.to_numpy()
y1_host -= y2_host
for i in range(y1_host.shape[0]):
    if(y1_host[i] != 0):
        print(f'prefix_sum error at {i}')
        break
print(y1_host)



# print(test_data)
# gpu_data = ti.field(ti.u32,shape=16)
# gpu_data.from_numpy(test_data)
# index_map = ti.field(ti.u32,shape=16)
# radix_sort(gpu_data,index_map)
# sorted_arr = ti.field(ti.u32,shape=16)

# for i in range(16):
#     sorted_arr[index_map[i]] = gpu_data[i]
# print(index_map.to_numpy())
# print(gpu_data.to_numpy())
# print(sorted_arr.to_numpy())


        