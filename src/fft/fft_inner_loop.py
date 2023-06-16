import numpy as np
import cv2
import taichi as ti
import taichi.math as tm


@ti.func
def log2_int(n: ti.u32):
    res = 0
    if n & ti.u32(0xffff0000):
        res += 16
        n >>= 16
    if n & ti.u32(0x0000ff00):
        res += 8
        n >>= 8
    if n & ti.u32(0x000000f0):
        res += 4
        n >>= 4
    if n & ti.u32(0x0000000c):
        res += 2
        n >>= 2
    if n & ti.u32(0x00000002):
        res += 1
        n >>= 1
    return res


def log2_int_(n: np.uint32):
    res = 0
    if n & np.uint32(0xffff0000):
        res += 16
        n >>= 16
    if n & np.uint32(0x0000ff00):
        res += 8
        n >>= 8
    if n & np.uint32(0x000000f0):
        res += 4
        n >>= 4
    if n & np.uint32(0x0000000c):
        res += 2
        n >>= 2
    if n & np.uint32(0x00000002):
        res += 1
        n >>= 1
    return res


@ti.func
def fast_mod(x, m):
    '''
    m must be the power of 2
    m is the power of 2
    '''
    return x & (m - 1)


@ti.func
def fast_div(x, log2_m):
    return x >> log2_m


@ti.func
def fast_mul(x, log2_m):
    '''
    m must be the power of 2
    m is the power of 2
    '''
    return x << log2_m


# @ti.func
# def vec4_cconj(z):
#     return tm.vec4(z.x, -z.y, z.z, -z.w)

@ti.func
def fft(x: ti.template(), buffer: ti.template(), inverse: ti.template()):
    log2_size = ti.static(log2_int_(x.shape[0]))

    size = ti.static(x.shape[0])
    ti.static_assert(size == 1 << log2_size, "size must be power of 2")
    ti.static_assert(
        size == buffer.shape[1], "buffer size must be equal to size")
    ti.static_assert(buffer.shape[0] == 2, "buffer must x2 size")
    ti.static_assert(buffer.dtype == x.dtype,
                     "buffer dtype must be equal to x")
    ti.static_assert(x.n == 2, "x must be complex")

    for i in x:
        buffer[0, i] = x[i]

    ti.block_local(buffer)

    target_flag: ti.i32 = 1
    ti.loop_config(block_dim=size)
    for k in x:
        tf = target_flag
        for s in ti.static(range(log2_size)):
            # print(f'k={k}, s={s}')
            b = size >> (s + 1)  # [4 2 1] # block size
            # opt w_ = (k // b_) * b_
            w = fast_div(k, log2_size - s - 1) * b  # overall rotation
            ev = fast_mod(w + k, size)
            od = ev + b
            phi = 2 * tm.pi / size * w

            twiddle = tm.vec2(tm.cos(phi), tm.sin(phi))
            if inverse:
                twiddle.y *= -1

            buffer[tf, k] = \
                buffer[not tf, ev] + \
                tm.cmul(buffer[not tf, od], twiddle)
            tf = not tf
            ti.simt.block.mem_sync()

    if log2_size & 1:
        target_flag = not target_flag

    if inverse:
        for i in x:
            x[i] = buffer[not target_flag, i] / size
    else:
        for i in x:
            x[i] = buffer[not target_flag, i]


@ti.kernel
def fft_forard(img: ti.template(), freq: ti.template(), buffer: ti.template()):
    size = img.shape[0]
    fft(img, buffer, False)

    for k in img:
        z1 = img[k]
        z2 = tm.cconj(img[size-k])
        freq[k] = tm.vec4(
            (z1 + z2)/2,
            (tm.cmul(tm.vec2(0, -0.5), z1 - z2)),
        )

@ti.kernel
def fft_inverse(freq: ti.template(), img: ti.template(), buffer: ti.template()):
    size = img.shape[0]
    for k in img:
        img[k] = freq[k].xy + tm.cmul(tm.vec2(0, 1), freq[k].zw)

    fft(img, buffer, True)

@ti.kernel
def direct_fft(x: ti.template(), buffer: ti.template(), inverse: ti.template()):
    fft(x, buffer, inverse)

ti.init(arch=ti.gpu)

size = 1 << 10
x = ti.Vector.field(2, dtype=ti.f32, shape=size)
freq = ti.Vector.field(4, dtype=ti.f32, shape=size)
buffer = ti.Vector.field(2, dtype=ti.f32, shape=(2, size))
x_np = np.random.randn(size).astype(np.float32)
y_np = np.random.randn(size).astype(np.float32)

data = np.stack([x_np, y_np], axis=0)
x.from_numpy(data.T)


x_t = ti.Vector.field(2, dtype=ti.f32, shape=size)
x_np = np.stack([x_np, np.zeros_like(x_np)], axis=0)
x_t.from_numpy(x_np.T)
direct_fft(x_t, buffer, False)

fft_forard(x, freq, buffer)
x.fill(0)
fft_inverse(freq, x, buffer)

print(np.max(np.abs(x.to_numpy().T - data)))

fft_res = freq.to_numpy().T
fft_res = fft_res[0:2, :]

print(np.max(np.abs(fft_res - x_t.to_numpy().T)))



# res_ = res.to_numpy().T

# # compare with numpy
# x_np = np.fft.fft(x_np)
# y_np = np.fft.fft(y_np)
# x_np = np.stack([x_np.real, x_np.imag], axis=0)
# y_np = np.stack([y_np.real, y_np.imag], axis=0)
# # stack y axis together
# res_np = np.stack([x_np, y_np], axis=0).reshape(4, -1)


# print(np.max(np.abs(res_ - res_np)))
