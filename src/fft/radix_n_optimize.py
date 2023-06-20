import numpy as np
import cv2
import taichi as ti
import taichi.math as tm
import time
import threading


def factorize(n, max_factor):
    def factor2(n):
        log2_factor = 0
        if n & 0x0000ffff == 0:
            log2_factor += 16
            n >>= 16
        if n & 0x000000ff == 0:
            log2_factor += 8
            n >>= 8
        if n & 0x0000000f == 0:
            log2_factor += 4
            n >>= 4
        if n & 0x00000003 == 0:
            log2_factor += 2
            n >>= 2
        if n & 0x00000001 == 0:
            log2_factor += 1
            n >>= 1
        return log2_factor

    log2_factor = factor2(n)
    n >>= log2_factor
    factors = []
    f = max_factor - 1
    sqrt_n = int(np.sqrt(n))
    while f >= sqrt_n:
        f -= 2
    while f > 2 and n > max_factor:
        if n % f == 0:
            factors.append(f)
            n //= f
        else:
            f -= 2
    if n != 1:
        if n > max_factor:
            f = 3
            while f*f < n:
                if n % f == 0:
                    factors.append(f)
                    n //= f
                else:
                    f += 2
            if n != 1:
                factors.append(n)
        else:
            factors.append(n)
    return log2_factor, factors


def radix_seqence(n, log2_max_radix):
    radix = []
    log2_radix = []
    max_radix = 1 << log2_max_radix
    log2_factor, factors = factorize(n, max_radix)

    for f in factors:
        radix.append(f)
        log2_radix.append(0)
    non_pow2_pass = len(radix)

    max_radix_pass = log2_factor//log2_max_radix
    for _ in range(max_radix_pass):
        radix.append(max_radix)
        log2_radix.append(log2_max_radix)
    pow2_pass = max_radix_pass
    log2_min_radix = log2_factor % log2_max_radix
    min_radix = 1 << log2_min_radix
    if log2_min_radix != 0:
        radix.append(min_radix)
        log2_radix.append(log2_min_radix)
        pow2_pass += 1

    radix_prod = [1 for i in radix]
    for i in range(len(radix)-2, -1, -1):
        radix_prod[i] = radix_prod[i+1] * radix[i+1]
    log_2_radix_prod = [int(np.log2(r)) for r in radix_prod]

    return (radix, log2_radix,
            radix_prod, log_2_radix_prod,
            non_pow2_pass, pow2_pass)


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


def log2_int_static(n: np.uint32):
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
    '''
    return x & (m - 1)


@ti.func
def fast_div(x, log2_m):
    '''
    m must be the power of 2
    '''
    return x >> log2_m


@ti.func
def fast_mul(x, log2_m):
    '''
    m must be the power of 2
    '''
    return x << log2_m

# for non power of 2


@ti.func
def fft_non_pow2_pass(k: ti.u32,
                      size: ti.template(),
                      radix: ti.u32,
                      radix_prod: ti.u32,
                      buffer: ti.template(),
                      target,
                      inverse: ti.template()):
    b = radix_prod
    w0 = k//b*b
    c = w0*radix - w0 + k
    offset = k//size*size  # multiple target offset
    i = c % size + offset
    res = buffer[not target, i]
    for m in range(1, radix):
        i = (c + m*b) % size + offset
        w = w0 * m
        phi = -2 * tm.pi / size * w

        twiddle = tm.vec2(tm.cos(phi), tm.sin(phi))
        if inverse:
            twiddle.y = -twiddle.y

        v = buffer[not target, i]
        res += tm.vec4(
            tm.cmul(v.xy, twiddle),
            tm.cmul(v.zw, twiddle))
    buffer[target, k] = res

@ti.func 
def vec4_arr_get(arr, i):
    return tm.vec4(arr[i,0], arr[i,1], arr[i,2], arr[i,3])

@ti.func
def vec4_arr_set(arr, i, v):
    arr[i,0] = v.x
    arr[i,1] = v.y
    arr[i,2] = v.z
    arr[i,3] = v.w

@ti.func
def print_radix_buffer(buffer:ti.template()):
    shape = buffer.get_shape()
    for i in range(shape[0]):
        for j in range(shape[1]):
            print(buffer[i,j], end=' ')
        print()

@ti.func
def fft_subpass(vec:ti.template(),
                k_:ti.template(),
                N_:ti.template(),
                P_:ti.template(),
                R_:ti.template(),
                inverse:ti.template()):
    shape = vec.get_shape()
    r_ofs = 0
    w_ofs = R_
    
    N = R_
    R = ti.static(int(2))
    P = N
    while P > 1:
        P //= R
        S = N//R
        for i in range(S):
            k,p = i//P, i%P
            src = ti.Vector([P*R*k + p + P*r for r in range(R)])
            dst = ti.Vector([i + t*S         for t in range(R)])
            # src_vec = ti.Vector([
            #     [vec[src[r] + r_ofs, x] for x in range(4)] for r in range(R)
            #     ])
            phi = -2 * tm.pi / N * (k * P)
            twiddle = tm.vec2(tm.cos(phi), tm.sin(phi))
            if inverse:
                twiddle.y = -twiddle.y 
            f1 = vec4_arr_get(vec, src[0] + r_ofs)
            f2 = vec4_arr_get(vec, src[1] + r_ofs)
            f2 = tm.vec4(tm.cmul(f2.xy, twiddle), tm.cmul(f2.zw, twiddle))
            v1 = f1 + f2
            v2 = f1 - f2
            for x in range(4):
                vec[dst[0] + w_ofs,x] = v1[x]
                vec[dst[1] + w_ofs,x] = v2[x]
        w_ofs,r_ofs = r_ofs,w_ofs

    return vec, r_ofs


@ti.func
def fft_pass(gtid: ti.u32,
             size: ti.template(),
             log2_size: ti.template(),
             is_pow2_size: ti.template(),
             radix: ti.template(),
             log2_radix: ti.template(),
             radix_prod: ti.u32,
             log_2_radix_prod: ti.u32,
             buffer: ti.template(),
             target,
             inverse: ti.template()):
    
    log2_P = log_2_radix_prod
    P = radix_prod
    R = ti.static(radix)
    offset = 0 # multiple target offset
    t_size = size//R
    log2_t_size = log2_size - log2_radix
    if ti.static(is_pow2_size):
        # offset = fast_mul(fast_mul(fast_div(gtid, log2_t_size), log2_t_size),log2_radix)
        offset = fast_mul(fast_div(fast_mul(gtid, log2_radix),log2_size), log2_size)
    else:
        offset = gtid//t_size * t_size * R
    #read
    radix_buffer = ti.Vector([[0 for x in range(4)]for r in range(2*R)],
                              dt=ti.f32)
    
    k,p = (gtid%t_size)//P, (gtid%t_size)%P
    for r in range(R):
        for x in range(4):
            radix_buffer[r,x] = buffer[not target, P*R*k + p + P*r + offset][x]
    
    if gtid == 0:
        print(f'---------P = {P}-- R = {R}----------')
    if gtid//t_size == 0:
        print(f'gtid = {gtid}, offset = {offset:3d}, '
            f'src = [{P*R*k + p + P*0 + offset:3d},{P*R*k + p + P*1 + offset:3d}], '
            f'dst = [{gtid%t_size + 0*t_size + offset:3d},{gtid%t_size + 1*t_size + offset:3d}], ')
        
    radix_buffer,ofs = fft_subpass(radix_buffer, k,size,P,R, inverse)
    

    for r in range(R):
        buffer[target,gtid%t_size + r*t_size + offset] = vec4_arr_get(radix_buffer, r + ofs)



@ti.func
def fft_grouped(size: ti.template(),
                pixel_size,
                inverse: ti.template(),
                buffer: ti.template(),
                target_):

    (radix, log2_radix,
     radix_prod, log_2_radix_prod,
     non_pow2_pass, pow2_pass) = ti.static(radix_seqence(size, 2))

    log2_size = ti.static(log2_int_static(size))
    is_pow2_size = ti.static(size & (size - 1) == 0)
    total_pass = ti.static(non_pow2_pass + pow2_pass)
        
    # ti.loop_config(block_dim=size)
    # for k in range(pixel_size):
    #     target = target_
    #     for step in range(0,non_pow2_pass):
    #         fft_non_pow2_pass(k, size,
    #                           radix[step],
    #                           radix_prod[step],
    #                           buffer, target, inverse)
    #         target = not target
    #         ti.simt.block.mem_sync()
    #     for step in range(non_pow2_pass,total_pass):
    #         fft_pass(k, size, log2_size, is_pow2_size,
    #                  radix[step], log2_radix[step],
    #                  radix_prod[step], log_2_radix_prod[step],
    #                  buffer, target, inverse)
    #         target = not target
    #         ti.simt.block.mem_sync()

   
    target = target_
    # for step in ti.static(range(0,non_pow2_pass)):
    #     ti.loop_config(block_dim=size)
    #     for k in range(pixel_size):
    #         fft_non_pow2_pass(k, size,
    #                             radix[step],
    #                             radix_prod[step],
    #                             buffer, target, inverse)
    #     target = not target
    for step in ti.static(range(non_pow2_pass,total_pass)):
        ti.loop_config(block_dim=size//radix[step])
        for gtid in range(pixel_size//radix[step]):
            fft_pass(gtid, size, log2_size, is_pow2_size,
                        radix[step], log2_radix[step],
                        radix_prod[step], log_2_radix_prod[step],
                        buffer, target, inverse)
        target = not target

    return ti.static(total_pass)


@ti.func
def fft_2d(xy: ti.template(), buffer: ti.template(), inverse: ti.template()):
    '''
    xy : [y_size, x_size][4] 2 complex
    buffer: [2, x_size * y_size][4] 2 * 2 complex
    '''
    x_size = ti.static(xy.shape[1])
    y_size = ti.static(xy.shape[0])
    log2_x_size = ti.static(log2_int_static(x_size))
    log2_y_size = ti.static(log2_int_static(y_size))
    is_pow2_x = ti.static(x_size & (x_size - 1) == 0)
    is_pow2_y = ti.static(y_size & (y_size - 1) == 0)

    pixel_size = ti.static(x_size * y_size)

    # ti.block_local(buffer)

    target: ti.i32 = 1
    # buffer copy
    for i, j in xy:
        index = 0
        if ti.static(is_pow2_x):
            index = fast_mul(i, log2_x_size) + j
        else:
            index = i*x_size + j
        buffer[0, index] = xy[i, j]

    # horizontal
    swap_count = fft_grouped(x_size, pixel_size, inverse, buffer, target)

    if fast_mod(swap_count, 2) == 1:
        target = not target

    # swap x,y
    ti.loop_config(block_dim=x_size)
    for i, j in xy:
        old_index = 0
        new_index = 0
        if ti.static(is_pow2_x):
            old_index = fast_mul(i, log2_x_size) + j
        else:
            old_index = i*x_size + j
        if ti.static(is_pow2_y):
            new_index = fast_mul(j, log2_y_size) + i
        else:
            new_index = j*y_size + i
        if inverse:
            buffer[target, new_index] = buffer[not target, old_index]/x_size
        else:
            buffer[target, new_index] = buffer[not target, old_index]
    target = not target

    # vertical
    swap_count = fft_grouped(y_size, pixel_size, inverse, buffer, target)

    if fast_mod(swap_count, 2) == 1:
        target = not target

    # copy back
    for i, j in xy:
        index = 0
        if ti.static(is_pow2_y):
            index = fast_mul(j, log2_y_size) + i
        else:
            index = j*y_size + i
        if inverse:
            xy[i, j] = buffer[not target, index] / y_size
        else:
            xy[i, j] = buffer[not target, index]


@ti.func
def vec4_cconj(vec: tm.vec4):
    return tm.vec4(vec.x, -vec.y, vec.z, -vec.w)


@ti.kernel
def quad_fft(img: ti.template(),
             freq: ti.template(),
             buffer: ti.template(),
             inverse: ti.template()):
    '''
    img: [size_y, size_x][4] rgba
    freq: [size_y, size_x, 2][4] (rfgf, bfaf)
    buffer: [2, size_y * size_x][4]
    '''
    n = ti.static(img.shape[0])
    m = ti.static(img.shape[1])
    if ti.static(not inverse):
        fft_2d(img, buffer, inverse=inverse)
        for i, j in img:
            z1 = img[i, j]
            rev_i = n-i
            rev_j = m-j
            if rev_i == n: rev_i = 0
            if rev_j == m: rev_j = 0
            z2 = vec4_cconj(img[rev_i, rev_j])
            rfgf = tm.vec4(0.5 * (z1.xy + z2.xy),
                           tm.cmul(tm.vec2(0, -0.5), z1.xy - z2.xy))
            bfaf = tm.vec4(0.5 * (z1.zw + z2.zw),
                           tm.cmul(tm.vec2(0, -0.5), z1.zw - z2.zw))
            freq[i, j, 0] = rfgf
            freq[i, j, 1] = bfaf
    else:
        for i, j in img:
            img[i, j].xy = freq[i, j, 0].xy + \
                tm.cmul(tm.vec2(0, 1), freq[i, j, 0].zw)
            img[i, j].zw = freq[i, j, 1].xy + \
                tm.cmul(tm.vec2(0, 1), freq[i, j, 1].zw)
        fft_2d(img, buffer, inverse=inverse)


# numpy replacement for quad_fft
def numpy_quad_fft(img, freq, buffer, inverse):
    np_img = img.to_numpy()
    [r, g, b, a] = np.squeeze(np.split(np_img, 4, axis=2))
    np_freq = freq.to_numpy()

    if not inverse:
        fft_res_rg = np.fft.fft2(r + 1j*g)
        fft_res_ba = np.fft.fft2(b + 1j*a)
        R = 1/2 * (fft_res_rg + np.conj(fft_res_rg[::-1, ::-1]))
        G = -1j/2 * (fft_res_rg - np.conj(fft_res_rg[::-1, ::-1]))
        B = 1/2 * (fft_res_ba + np.conj(fft_res_ba[::-1, ::-1]))
        A = -1j/2 * (fft_res_ba - np.conj(fft_res_ba[::-1, ::-1]))
        np_freq[:, :, 0, 0] = R.real
        np_freq[:, :, 0, 1] = R.imag
        np_freq[:, :, 0, 2] = G.real
        np_freq[:, :, 0, 3] = G.imag
        np_freq[:, :, 1, 0] = B.real
        np_freq[:, :, 1, 1] = B.imag
        np_freq[:, :, 1, 2] = A.real
        np_freq[:, :, 1, 3] = A.imag
    else:
        rg_ifft_in = np_freq[:, :, 0, 0] + np_freq[:, :, 0, 1]*1j + \
            1j*(np_freq[:, :, 0, 2] + np_freq[:, :, 0, 3]*1j)
        ba_ifft_in = np_freq[:, :, 1, 0] + np_freq[:, :, 1, 1]*1j + \
            1j*(np_freq[:, :, 1, 2] + np_freq[:, :, 1, 3]*1j)
        fft_res_rg = np.fft.ifft2(rg_ifft_in)
        fft_res_ba = np.fft.ifft2(ba_ifft_in)
        np_img[:, :, 0] = fft_res_rg.real
        np_img[:, :, 1] = fft_res_rg.imag
        np_img[:, :, 2] = fft_res_ba.real
        np_img[:, :, 3] = fft_res_ba.imag
    img.from_numpy(np_img)
    freq.from_numpy(np_freq)


@ti.kernel
def compute_frequency_filter(frequency_filter: ti.template(),
                             rgba_buffer: ti.template(),
                             buffer: ti.template(),
                             sigma_x: ti.f32,
                             sigma_y: ti.f32):
    size_y, size_x = frequency_filter.shape
    for i, j in frequency_filter:
        y_distance = (i + size_y // 2) % size_y - size_y // 2
        x_distance = (j + size_x // 2) % size_x - size_x // 2
        sigma_xy = sigma_x * sigma_y
        a = 2 * np.pi * sigma_xy
        x_s = x_distance * sigma_x
        x_s2 = x_s * x_s
        y_s = y_distance * sigma_y
        y_s2 = y_s * y_s
        b = -0.5 * (x_s2 + y_s2)/(sigma_xy * sigma_xy)
        val = tm.exp(b) / a
        rgba_buffer[i, j] = tm.vec4(val, 0, 0, 0)
    fft_2d(rgba_buffer, buffer, inverse=False)
    for i, j in frequency_filter:
        frequency_filter[i, j] = rgba_buffer[i, j].xy


@ti.func
def luminance(c: ti.template()):
    return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z
    # b = 0.4
    # g = 0.4
    # r = 1 - b - g
    # return  r * c.x + g * c.y + b * c.z


@ti.kernel
def HDR_threshold(img: ti.template(), threshold: ti.f32):
    for i, j in img:
        luma = luminance(img[i, j])
        scaler = tm.pow(luma, threshold)
        img[i, j] = img[i, j] * scaler


@ti.kernel
def image_mul(freq: ti.template(), frequency_filter: ti.template(), scale: ti.f32):
    '''
    freq: [size_y, size_x, 2][4] (rfgf, bfaf)
    frequency_filter: [size_y, size_x][2] (rf)
    '''
    for i, j in frequency_filter:
        freq[i, j, 0].xy = tm.cmul(
            freq[i, j, 0].xy, frequency_filter[i, j]) * scale
        freq[i, j, 0].zw = tm.cmul(
            freq[i, j, 0].zw, frequency_filter[i, j]) * scale
        freq[i, j, 1].xy = tm.cmul(
            freq[i, j, 1].xy, frequency_filter[i, j]) * scale
        freq[i, j, 1].zw = tm.cmul(
            freq[i, j, 1].zw, frequency_filter[i, j]) * scale


@ti.kernel
def image_blend(src: ti.template(), target: ti.template()):
    for i, j in target:
        src_i = ti.i32(i * src.shape[0] / target.shape[0])
        src_j = ti.i32(j * src.shape[1] / target.shape[1])
        if target.dtype == ti.u8:
            target[i, j] = min(
                target[i, j] + min(src[src_i, src_j]*256, 255), 255)
        else:
            # linear add
            target[i, j] = target[i, j] + src[src_i, src_j]


@ti.kernel
def gamma_correction(img: ti.template(), gamma: ti.f32):
    for i, j in img:
        if img.dtype == ti.u8:
            img[i, j] = ti.pow(img[i, j] / 256, gamma) * 256
        else:
            img[i, j] = ti.pow(img[i, j], gamma)


@ ti.kernel
def ACES_tonemap(img: ti.template(), adapted_luminance: ti.f32):
    for i, j in img:
        if img.dtype == ti.u8:
            c = img[i, j] / 256 * adapted_luminance
            img[i, j] = (c * (2.51 * c + 0.03)) / \
                (c * (2.43 * c + 0.59) + 0.14) * 256
        else:
            c = img[i, j] * adapted_luminance
            img[i, j] = (c * (2.51 * c + 0.03)) / \
                (c * (2.43 * c + 0.59) + 0.14)


@ti.kernel
def downsample(src: ti.template(), dst: ti.template()):
    src_x, src_y = src.shape
    dst_x, dst_y = dst.shape
    for i, j in dst:
        src_i = i * src_x // dst_x
        src_j = j * src_y // dst_y
        if src.dtype == ti.u8:
            dst[i, j] = src[src_i, src_j] / 256
        else:
            dst[i, j] = src[src_i, src_j]


@ti.func
def bilinear_interpolation(src: ti.template(), u: ti.f32, v: ti.f32):
    x = (u - ti.floor(u)) * src.shape[0]
    y = (v - ti.floor(v)) * src.shape[1]
    x0 = ti.i32(ti.floor(u))
    x1 = ti.i32(x0 + 1)
    y0 = ti.i32(ti.floor(v))
    y1 = ti.i32(y0 + 1)
    x0 = ti.max(x0, 0)
    x1 = ti.min(x1, src.shape[0] - 1)
    y0 = ti.max(y0, 0)
    y1 = ti.min(y1, src.shape[1] - 1)
    x = x - x0
    y = y - y0
    return (1 - x) * (1 - y) * src[x0, y0] + (1 - x) * y * src[x0, y1] + x * (1 - y) * src[x1, y0] + x * y * src[x1, y1]


@ti.kernel
def field_transpose(src: ti.template(), dst: ti.template()):
    if dst.dtype == ti.u8 and (src.dtype == ti.f32 or src.dtype == ti.f16):
        for i, j in dst:
            dst[i, j] = ti.min(src[j * src.shape[0] // dst.shape[1],
                               i * src.shape[1] // dst.shape[0]] * 256, 255)
    else:
        for i, j in dst:
            dst[i, j] = src[j * src.shape[0] // dst.shape[1],
                            i * src.shape[1] // dst.shape[0]]


@ti.kernel
def freq_visualize(freq: ti.template(), transposed: ti.template(),scale_factor: ti.f32):
    '''
    freq: [size_y, size_x, 2][4] (rfgf, bfaf)
    transposed: [size_x, size_y][4] (rgba)
    buffer: [2, size_y * size_x][4]
    '''
    for i, j in transposed:
        i_ = (i + transposed.shape[0] // 2) % transposed.shape[0]
        j_ = (j + transposed.shape[1] // 2) % transposed.shape[1]
        i_freq = j * freq.shape[0] // transposed.shape[1]
        j_freq = i * freq.shape[1] // transposed.shape[0]
        transposed[i_, j_].xy = tm.min(
            tm.log2(tm.vec2(tm.length(freq[i_freq, j_freq, 0].xy),
                            tm.length(freq[i_freq, j_freq, 0].zw)) + 1)/scale_factor*256,
                            255)
        transposed[i_, j_].zw = tm.min(
            tm.log2(tm.vec2(tm.length(freq[i_freq, j_freq, 1].xy),
                            tm.length(freq[i_freq, j_freq, 1].zw)) + 1)/scale_factor*256,
                            255)


def main(kernel_profiler=False, debug=False):

    # fft_size_x = 1920
    # fft_size_y = 1080
    # img_size_x = 1920
    # img_size_y = 1080

    # fft_size_x = 3840//4
    # fft_size_y = 2048//4
    # img_size_x = 3840//2
    # img_size_y = 2048//2
    fft_size_x = 1<<3
    fft_size_y = 1<<3
    img_size_x = 1<<11
    img_size_y = 1<<10

    img_cv = cv2.imread(
        r'C:/Users/Estelle/source/repos/TaichiSandBox/src/fft/1.png', cv2.IMREAD_COLOR)
    img_cv = cv2.resize(img_cv, (img_size_x, img_size_y))
    # reorganize the image as rgb and inverse the y axis
    img_cv = img_cv[::-1, :, ::-1]

    # add alpha channel
    img_cv = np.concatenate(
        (img_cv, np.ones((img_size_y, img_size_x, 1), dtype=np.float32)), axis=2)
    img_cv = img_cv.astype(np.float16)

    ti.init(arch=ti.cuda, kernel_profiler=kernel_profiler, debug=debug)

    # load to ti
    original = ti.Vector.field(4, dtype=ti.u8, shape=(img_size_y, img_size_x))
    render_taget = ti.Vector.field(
        4, dtype=ti.u8, shape=(img_size_y, img_size_x))
    transposed = ti.Vector.field(
        4, dtype=ti.u8, shape=(img_size_x, img_size_y))
    # fft
    rgba = ti.Vector.field(4, dtype=ti.f32, shape=(fft_size_y, fft_size_x))
    rgba_freq = ti.Vector.field(
        4, dtype=ti.f32, shape=(fft_size_y, fft_size_x, 2))
    buffer = ti.Vector.field(
        4, dtype=ti.f32, shape=(2, fft_size_x * fft_size_y))
    # for blur the image
    frequency_fiter = ti.Vector.field(
        2, dtype=ti.f32, shape=(fft_size_y, fft_size_x))
    frequency_fiter.fill(1)

    original.from_numpy(img_cv)

    lock_wait_data = threading.Lock()
    lock_wait_data_transmit = threading.Lock()
    lock_wait_data.acquire()

    bloom_on = True
    kernel_profiler_print = False
    bloom_threshold = 10
    bloom_intensity = 1
    filter_x_scale = 4
    filter_y_scale = 20
    gamma = 1.2
    adaptive_luminance = 1

    show_freq = False
    use_ACES_tonemapping = False
    use_Gamma_correction = False

    use_numpy_fft = False

    program_running = True
    restart = False

    total_time = 0
    wait_transmit_time = 0
    process_time = 0
    fft_time = 0
    ifft_time = 0
    image_transpose_time = 0
    image_copy_time = 0
    present_time = 0
    before_fft_time = 0
    between_fft_ifft_time = 0
    after_ifft_time = 0
    profile_extime = 0
    freq_visualize_factor = 4

    def weighted_average(new, old):
        return new * 0.01 + old * 0.99

    def gui_thread():
        nonlocal program_running
        nonlocal bloom_on
        nonlocal bloom_threshold
        nonlocal bloom_intensity
        nonlocal filter_x_scale
        nonlocal filter_y_scale
        nonlocal gamma
        nonlocal adaptive_luminance
        nonlocal kernel_profiler
        nonlocal kernel_profiler_print
        nonlocal use_numpy_fft
        nonlocal image_copy_time
        nonlocal present_time
        nonlocal show_freq
        nonlocal use_ACES_tonemapping
        nonlocal use_Gamma_correction
        nonlocal restart
        nonlocal freq_visualize_factor

        window = ti.ui.Window('Real Time FFT Image Convolution',
                              (img_size_x, img_size_y), fps_limit=200,)
        gui = window.get_gui()
        canvas = window.get_canvas()

        fft_x_dir_radix,_,_,_,_,_ = radix_seqence(fft_size_x,2)
        fft_y_dir_radix,_,_,_,_,_ = radix_seqence(fft_size_y,2)

        while window.running:
            # t0 = time.time()
            if window.get_event(ti.ui.PRESS):
                if window.event.key in [ti.ui.ESCAPE]:
                    break

            with gui.sub_window("GUI", x=0.01, y=0.01, width=0.25, height=0.6):
                filter_x_scale = gui.slider_float("filter_x_scale", filter_x_scale, 0.5, 200)
                filter_y_scale = gui.slider_float("filter_y_scale", filter_y_scale, 0.5, 200)
                bloom_threshold = gui.slider_float("bloom_threshold", bloom_threshold, 0.0, 20.0)
                bloom_intensity = gui.slider_float("bloom_intensity", bloom_intensity, 0.0, 10.0)
                bloom_on = gui.checkbox("bloom", bloom_on)
                use_numpy_fft = gui.checkbox("numpy FFT", use_numpy_fft)
                show_freq = gui.checkbox("show frequency", show_freq)
                if show_freq:
                    freq_visualize_factor = gui.slider_float("freq_visualize_factor", freq_visualize_factor, 1.0, 20.0)
                use_ACES_tonemapping = gui.checkbox("ACES tonemapping", use_ACES_tonemapping)
                if use_ACES_tonemapping:
                    adaptive_luminance = gui.slider_float("adaptive_luminance", adaptive_luminance, 0.0, 2.0)
                use_Gamma_correction = gui.checkbox("Gamma correction", use_Gamma_correction)
                if use_Gamma_correction:
                    gamma = gui.slider_float("gamma", gamma, 0.0, 2.0)

                gui.text(f'original image size:     {img_size_x}x{img_size_y}')
                gui.text(f'down sampled image size: {fft_size_x}x{fft_size_y}')
                gui.text(f'FFT Radix: {fft_x_dir_radix}x{fft_y_dir_radix}')

                if gui.checkbox("kernel profiler", kernel_profiler):
                    if kernel_profiler == False:
                        kernel_profiler = True
                        restart = True
                        break
                    kernel_profiler_print = gui.button("kernel profiler print")
                    gui.text(f'profile external time: {profile_extime * 1000:.2f} ms')
                else:
                    if kernel_profiler == True:
                        kernel_profiler = False
                        restart = True
                        break

                gui.text(f'\nresult could be inaccurate if not using kernel profiler\n'
                         f'processing time: {process_time * 1000:.2f} ms')
                gui.text(f'     [before  time: {before_fft_time*1000:.2f} ms]')
                gui.text(f'     [FFT     time: {fft_time*1000:.2f} ms]')
                gui.text(f'     [between time: {between_fft_ifft_time*1000:.2f} ms]')
                gui.text(f'     [IFFT    time: {ifft_time*1000:.2f} ms]')
                gui.text(f'     [after   time: {after_ifft_time*1000:.2f} ms]')
                gui.text(f'wait image transmit: {wait_transmit_time * 1000:.2f} ms')
                gui.text(f'transpose time: {image_transpose_time * 1000:.2f} ms')
                gui.text(f'transpose + set image: {image_copy_time * 1000:.2f} ms')
                gui.text(f'present time: {present_time * 1000:.2f} ms')
                gui.text(f'main thread time: {total_time * 1000:.2f} ms')

            t1 = time.time()
            lock_wait_data.acquire()
            canvas.set_image(transposed)
            lock_wait_data_transmit.release()
            t2 = time.time()
            window.show()
            t3 = time.time()
            image_copy_time = weighted_average(
                image_transpose_time + t2-t1, image_copy_time)
            present_time = weighted_average(t3-t2, present_time)

        program_running = False

    try:
        threading.Thread(target=gui_thread).start()

        (old_filter_x_scale, old_filter_y_scale) = (0, 0)
        while program_running:

            t0 = time.time()

            if kernel_profiler:
                lock_wait_data_transmit.acquire()
                lock_wait_data_transmit.release()

                ti.sync()
                ti.profiler.clear_kernel_profiler_info()

            t1 = time.time()
            profile_extime = weighted_average(t1-t0, profile_extime)

            # if old_filter_x_scale != filter_x_scale or old_filter_y_scale != filter_y_scale:
            #     compute_frequency_filter(
            #         frequency_fiter, rgba, buffer, filter_x_scale, filter_y_scale)
            #     old_filter_x_scale, old_filter_y_scale = filter_x_scale, filter_y_scale

            render_taget.copy_from(original)
            downsample(original, rgba)
            # HDR_threshold(rgba, bloom_threshold)

            t_f0 = time.time()
            before_fft_time = weighted_average(t_f0 - t1, before_fft_time)
            if use_numpy_fft:
                numpy_quad_fft(rgba, rgba_freq, buffer, False)
            else:
                quad_fft(rgba, rgba_freq, buffer, False)
            t_f1 = time.time()
            fft_time = weighted_average(t_f1 - t_f0, fft_time)

            # if bloom_on:
            #     image_mul(rgba_freq, frequency_fiter, bloom_intensity)

            t_f2 = time.time()
            between_fft_ifft_time = weighted_average(
                t_f2 - t_f1, between_fft_ifft_time)
            if use_numpy_fft:
                numpy_quad_fft(rgba, rgba_freq, buffer, True)
            else:
                quad_fft(rgba, rgba_freq, buffer, True)
            t_f3 = time.time()
            ifft_time = weighted_average(t_f3-t_f2, ifft_time)

            # image_blend(rgba, render_taget)

            if use_ACES_tonemapping:
                ACES_tonemap(render_taget, adaptive_luminance)
            if use_Gamma_correction:
                gamma_correction(render_taget, gamma)

            if kernel_profiler_print:
                ti.profiler.print_kernel_profiler_info('trace')
                ti.profiler.print_kernel_profiler_info()

            t2 = time.time()
            after_ifft_time = weighted_average(t2 - t_f3, after_ifft_time)
            process_time = weighted_average(t2-t1, process_time)

            # wait for gui thread to go
            lock_wait_data_transmit.acquire()

            t3 = time.time()
            wait_transmit_time = weighted_average(t3-t2, wait_transmit_time)

            if show_freq:
                freq_visualize(rgba_freq, transposed, freq_visualize_factor)
            else:
                field_transpose(rgba, transposed)

            # if bloom_on:
            #     if show_freq:
            #         freq_visualize(rgba_freq, transposed, freq_visualize_factor)
            #     else:
            #         field_transpose(render_taget, transposed)
            # else:
            #     field_transpose(original, transposed)
            lock_wait_data.release()

            t4 = time.time()
            image_transpose_time = t4-t3

            total_time = weighted_average(t4-t0, total_time)

        return (restart, kernel_profiler, debug)
    except:
        raise


if __name__ == '__main__':
    kernel_profiler = True
    debug = False
    restart = True
    while restart:
        restart, kernel_profiler, debug = main(kernel_profiler, debug)
