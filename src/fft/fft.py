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


@ti.func
def bit_reverse(x: ti.i32, log2_size: ti.i32):
    res = 0
    for _ in range(log2_size):
        res <<= 1
        res |= x & 1
        x >>= 1
    return res

# @ti.func
# def bit_reverse(x: ti.i32, log2_size: ti.i32):
#     # Lookup table for bit-reversed indices
#     table = tm.ivec4(0, 2, 1, 3)

#     res = 0
#     for i in range(fast_div(log2_size, 1)):
#         t = (x & 3)
#         res = (res << 2) | table[t & 0b11]
#         x >>= 2
#     if log2_size & 1:
#         res = (res << 1) | (x & 0b1)
#     return res


# worse than fft
@ti.kernel
def fft_bit_reverse(x: ti.template(), buffer: ti.template(), inverse: ti.template()):
    size = ti.static(x.shape[0])
    log2_size = ti.static(log2_int_(x.shape[0]))

    assert size == 1 << log2_size, "size must be power of 2"
    assert size == buffer.shape[1], "buffer size must be equal to size"

    # bit reverse
    for i in x:
        buffer[0, bit_reverse(i, log2_size)] = x[i]

    target_flag: ti.i32 = 1
    # static loop expansion
    for s in ti.static(range(log2_size)):
        # parallel loop
        # print(
        #     f'----------s = {s}----------target_flag = {target_flag}----------')
        for k in x:
            b = 1 << (s+1)
            c = 1 << s
            bi = fast_mod(k, b)  # bi = k % b
            # ev = (k // b) * b + bi % (b//2)
            ev = fast_div(k, s+1) * b + fast_mod(bi, c)
            od = ev + c
            phi = -2 * tm.pi / b * bi

            twiddle = tm.vec2(tm.cos(phi), tm.sin(phi))
            if inverse:
                twiddle.y *= -1

            buffer[target_flag, k] = \
                buffer[not target_flag, ev] + \
                tm.cmul(buffer[not target_flag, od], twiddle)

        target_flag = not target_flag
    if inverse:
        for i in x:
            x[i] = buffer[not target_flag, i] / size
    else:
        for i in x:
            x[i] = buffer[not target_flag, i]


@ti.kernel
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

    target_flag: ti.i32 = 1
    # static loop expansion
    for s in ti.static(range(log2_size)):
        # parallel loop
        for k in x:

            b = size >> (s + 1)  # [4 2 1] # block size
            # opt w_ = (k // b_) * b_
            w = fast_div(k, log2_size - s - 1) * b  # overall rotation
            ev = fast_mod(w + k, size)
            od = ev + b
            phi = -2 * tm.pi / size * w

            twiddle = tm.vec2(tm.cos(phi), tm.sin(phi))
            if inverse:
                twiddle.y *= -1

            buffer[target_flag, k] = \
                buffer[not target_flag, ev] + \
                tm.cmul(buffer[not target_flag, od], twiddle)

        target_flag = not target_flag
    if inverse:
        for i in x:
            x[i] = buffer[not target_flag, i] / size
    else:
        for i in x:
            x[i] = buffer[not target_flag, i]


@ti.func
def print_complex_img(img: ti.template()):
    for i, j in img:
        if j != img.shape[1] - 1:
            print(f'{i}{j}[{img[i, j].x:.2f},{img[i, j].y:.2f}]', end=' ')
        else:
            print(f'{i}{j}[{img[i, j].x:.2f},{img[i, j].y:.2f}]')
    print()


@ti.func
def print_buffer(buffer: ti.template(), flag, size_x):
    for i in range(buffer.shape[1]):
        x_coord = fast_mod(i, size_x)
        y_coord = fast_div(i, log2_int(size_x))
        if x_coord != size_x - 1:
            print(
                f'{y_coord}{x_coord}[{buffer[flag,i].x:.2f},{buffer[flag,i].y:.2f}]', end=' ')
        else:
            print(
                f'{y_coord}{x_coord}[{buffer[flag,i].x:.2f},{buffer[flag,i].y:.2f}]')
    print()


debug = False


@ti.func
def fft_pass(s,
             size,
             log2_size,
             pixel_size,
             log2_pixel_size,
             inverse: ti.template(),
             buffer: ti.template(),
             target_flag):
    for i in range(pixel_size):
        # k = fast_mod(i, size)

        b = size >> (s + 1)  # [4 2 1] # block size
        log2_b = log2_size - s - 1
        # opt w_ = (k // b_) * b_
        # overall rotation #begin offset
        w = fast_mul(fast_div(i, log2_b), log2_b)
        ev = fast_mod(w + i, size) + \
            fast_mul(fast_div(i, log2_size), log2_size)
        od = ev + b
        phi = -2 * tm.pi / size * w

        if debug:
            print(
                f'i={i:.2i} b = {b:.2i} w = {w:.2i} ev = {ev:.2i} of = {fast_mul(fast_div(i, log2_size) ,log2_size):.2i} od = {od:.2i} phi/pi = {tm.mod(phi/tm.pi,2):.2f}')

        twiddle = tm.vec2(tm.cos(phi), tm.sin(phi))
        if inverse:
            twiddle.y = -twiddle.y

        buffer[target_flag, i] = \
            buffer[not target_flag, ev] + \
            tm.cmul(buffer[not target_flag, od], twiddle)


@ti.func
def fft_2d(img: ti.template(), buffer: ti.template(), inverse: ti.template()):

    x_size = ti.static(img.shape[1])
    y_size = ti.static(img.shape[0])
    log2_x_size = ti.static(log2_int_(x_size))
    log2_y_size = ti.static(log2_int_(y_size))

    pixel_size = ti.static(x_size * y_size)
    log2_pixel_size = ti.static(log2_int_(pixel_size))

    target_flag: ti.i32 = 1
    for i, j in img:
        index = fast_mul(i, log2_x_size) + j
        buffer[0, index] = img[i, j]
    if debug:
        print_complex_img(img)
        print(f'nontarget_flag={not target_flag}')
        print_buffer(buffer, (not target_flag), x_size)

    # horizontal
    for s in ti.static(range(log2_x_size)):
        if debug:
            print(f'-----------------s={s}-----------------')
            print(f'target_flag={target_flag}')
            print_buffer(buffer, target_flag, x_size)
            print(f'nontarget_flag={not target_flag}')
            print_buffer(buffer, (not target_flag), x_size)
        fft_pass(s, x_size, log2_x_size, pixel_size, log2_pixel_size,
                 inverse, buffer, target_flag)

        target_flag = not target_flag

    if debug:
        for i, j in img:
            index = fast_mul(i, log2_x_size) + j
            if inverse:
                img[i, j] = buffer[not target_flag, index]/x_size
            else:
                img[i, j] = buffer[not target_flag, index]

    if debug:
        print_complex_img(img)

    # swap x,y
    for i, j in img:
        old_index = fast_mul(i, log2_x_size) + j
        new_index = fast_mul(j, log2_y_size) + i
        if inverse:
            buffer[target_flag, new_index] = buffer[not target_flag,
                                                    old_index]/size_x
        else:
            buffer[target_flag, new_index] = buffer[not target_flag, old_index]
    target_flag = not target_flag

    # vertical
    for s in ti.static(range(log2_y_size)):
        if debug:
            print(f'-----------------s={s}-----------------')
            print(f'target_flag={target_flag}')
            print_buffer(buffer, target_flag, x_size)
            print(f'nontarget_flag={not target_flag}')
            print_buffer(buffer, (not target_flag), x_size)
        fft_pass(s, y_size, log2_y_size, pixel_size, log2_pixel_size,
                 inverse, buffer, target_flag)

        target_flag = not target_flag

    # copy back
    for i, j in img:
        index = fast_mul(j, log2_y_size) + i
        if inverse:
            img[i, j] = buffer[not target_flag, index]/size_y
        else:
            img[i, j] = buffer[not target_flag, index]


ti.init(arch=ti.vulkan)

size_x = 1 << 10
size_y = 1 << 9
if debug:
    size_x = 1 << 3
    size_y = 1 << 2


@ti.kernel
def img_to_complex(rgb: ti.template(), rgb2: ti.template()):
    for i, j in rgb:
        for c in ti.static(range(3)):
            rgb2[i, j, 0][c] = rgb[i, j][c]
            rgb2[i, j, 1][c] = 0


@ti.kernel
def complex_to_img(rgb2: ti.template(), rgb: ti.template()):
    for i, j in rgb:
        for c in ti.static(range(3)):
            rgb[i, j][c] = rgb2[i, j, 0][c]


img = ti.field(dtype=ti.f32, shape=(size_y, size_x))
buffer = ti.Vector.field(2, dtype=ti.f32, shape=(2, size_x * size_y))
complex_img = ti.Vector.field(2, dtype=ti.f32, shape=(size_y, size_x))

@ti.kernel
def image_fft(complex_rgb: ti.template(), inverse: ti.template()):
    for c in ti.static(range(3)):
        for i, j ,k in complex_rgb:
            complex_img[i,j][k] = complex_rgb[i,j,k][c]
        fft_2d(complex_img, buffer, inverse=inverse)
        for i,j,k in complex_rgb:
            complex_rgb[i,j,k][c] = complex_img[i,j][k]


@ti.kernel
def field_transpose(src: ti.template(), dst: ti.template()):
    for i, j in src:
        dst[j, i] = src[i, j]

#for blur the image
frequency_fiter = ti.field(dtype=ti.f32, shape=(size_y, size_x))
frequency_fiter.fill(1)
@ti.kernel
def compute_frequency_filter(frequency_filter: ti.template(),x_scale:ti.f32,y_scale:ti.f32):
    for i, j in frequency_filter:
        y_distance = (i + size_y // 2) % size_y - size_y // 2
        x_distance = (j + size_x // 2) % size_x - size_x // 2
        distance = ti.sqrt(x_distance * x_distance * x_scale + y_distance * y_distance * y_scale)
        frequency_filter[i, j] = ti.exp(-distance / 2)    

@ti.kernel
def image_blur(complex_img: ti.template(), frequency_filter: ti.template()):
    for i, j ,k in complex_img:
        for c in ti.static(range(3)):
            complex_img[i,j,k][c] *= frequency_filter[i,j]

def main():
    img_cv = cv2.imread(
        r'C:/Users/Estelle/source/repos/TaichiSandBox/src/fft/scr.jpg', cv2.IMREAD_COLOR)
    img_cv = cv2.resize(img_cv, (size_x, size_y))
    img_cv = img_cv.astype(np.float32)
    img_cv = img_cv / 255.0
    # reorganize the image as rgb and inverse the y axis
    img_cv = img_cv[::-1, :, ::-1]

    # load to ti
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(size_y, size_x))
    pixels_complex = ti.Vector.field(3, dtype=ti.f32, shape=(size_y, size_x, 2))
    # min_max_pixel = ti.Vector.field(3, dtype=ti.f32, shape=(size_y, size_x))

    original = ti.Vector.field(3, dtype=ti.f32, shape=(size_y, size_x))
    original.from_numpy(img_cv)
    pixels.from_numpy(img_cv)
    transposed = ti.Vector.field(3, dtype=ti.f32, shape=(size_x, size_y))

    # #test for single channel
    # test_img = img_cv[:,:,0]
    # test_img = test_img.reshape((size_y,size_x))
    # cv2.imshow('test',abs(test_img))

    # #load to ti
    # @ti.kernel
    # def img_to_complex_test(img:ti.template(),complex_img:ti.template()):
    #     img_to_complex(img,complex_img)

    # @ti.kernel
    # def complex_to_img_test(complex_img:ti.template(),img:ti.template()):
    #     complex_to_img(complex_img,img)

    # @ti.kernel
    # def fft_2d_test(complex_img:ti.template(),buffer:ti.template(),inverse:ti.template()):
    #     fft_2d(complex_img,buffer,inverse)

    # test_img_d = ti.field(dtype=ti.f32, shape=(size_y, size_x))
    # test_img_d.from_numpy(test_img)
    # img_to_complex_test(test_img_d,complex_img)
    # fft_2d_test(complex_img,buffer,inverse=False)
    # complex_to_img_test(complex_img,test_img_d)
    # test_img = test_img_d.to_numpy()
    # #show in cv
    # cv2.imshow('test',test_img)
    # test_img_d.from_numpy(test_img)
    # img_to_complex_test(test_img_d,complex_img)
    # #inverse
    # fft_2d_test(complex_img,buffer,inverse=True)
    # complex_to_img_test(complex_img,test_img_d)
    # test_img = test_img_d.to_numpy()
    # cv2.imshow('test2',test_img)
    # cv2.waitKey(0)

    # # using np.fft
    # test_img = np.fft.fft2(test_img)
    # cv2.imshow('test3',abs(test_img))
    # test_img = np.fft.ifft2(test_img)
    # cv2.imshow('test4',abs(test_img))
    # cv2.waitKey(0)

    gui = ti.GUI("FFT", (size_x, size_y))
    filter_x_scale = gui.slider("filter_x_scale", 0.0, 1.0, 0.5)
    filter_y_scale = gui.slider("filter_y_scale", 0.0, 1.0, 0.5)

    try:
        while gui.running:

                # options for original\fft\ifft
                for e in gui.get_events(ti.GUI.PRESS):
                    if e.key == ti.GUI.ESCAPE:
                        gui.running = False                
                        
                #detect the change of the slider

                compute_frequency_filter(frequency_fiter, filter_x_scale.value, filter_y_scale.value)
                # cv2.imshow("frequency_filter", np.fft.fftshift(frequency_fiter.to_numpy()))

                img_to_complex(original, pixels_complex)
                image_fft(pixels_complex, inverse=False)
                image_blur(pixels_complex, frequency_fiter)
                image_fft(pixels_complex, inverse=True)
                complex_to_img(pixels_complex, pixels)

                field_transpose(pixels, transposed)
                gui.set_image(transposed)
                gui.show()
    except:
        gui.close()
        cv2.destroyAllWindows()
        raise
    cv2.destroyAllWindows()
    gui.close()


if __name__ == '__main__':
    main()


# buffer = ti.Vector.field(2, dtype=ti.f32, shape=(2, size))
# # test
# # complex number
# x = np.random.randn(2, size).astype(np.float32)
# x_d = ti.Vector.field(2, dtype=ti.f32, shape=size)
# x_d.from_numpy(x.T)
# print(f'host x = {x}')

# fft(x_d, buffer, inverse=False)

# x_res = x_d.to_numpy().T
# x_np = np.fft.fft(x[0]+1j*x[1])

# np.set_printoptions(precision=2, suppress=True)
# print(f'x_res = {x_res}')
# print(f'x_np = {x_np}')
# print(f'diff = {np.sum(x_res[0]+1j*x_res[1] - x_np)}')

# fft(x_d, buffer, inverse=True)

# x_res = x_d.to_numpy().T
# x_np = np.fft.ifft(x_np)
# print("inverse")
# print(f'x_res = {x_res}')
# print(f'x_np = {x_np}')
# print(f'diff = {np.sum(x_res[0]+1j*x_res[1] - x_np)}')


# # compare presort_fft and fft and np.fft.fft

# x_d.from_numpy(x.T)

# ti.profiler.clear_kernel_profiler_info()  # Clears all records
# fft(x_d, buffer, inverse=False)
# ti.profiler.print_kernel_profiler_info()
# x_d.from_numpy(x.T)
# ti.profiler.clear_kernel_profiler_info()  # Clears all records
# presort_fft(x_d, buffer, inverse=False)
# ti.profiler.print_kernel_profiler_info()
