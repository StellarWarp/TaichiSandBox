import taichi as ti

@ti.func
def expand_bits_3d(x: ti.u32):
    # expand 10-bit integer to 30-bit by inserting 2 zeros after each bit
    # x: the 10-bit integer
    # return: the 30-bit integer
    x = (x * ti.u32(0x00010001)) & ti.u32(0xFF0000FF)
    x = (x * ti.u32(0x00000101)) & ti.u32(0x0F00F00F)
    x = (x * ti.u32(0x00000011)) & ti.u32(0xC30C30C3)
    x = (x * ti.u32(0x00000005)) & ti.u32(0x49249249)
    return x

@ti.func
def expand_bits_2d(x:ti.u32):
    # expand 10-bit integer to 20-bit by inserting 1 zero after each bit
    # x: the 10-bit integer
    # return: the 20-bit integer
    x = (x | (x << 16)) & ti.u32(0x0000FFFF)
    x = (x | (x << 8)) & ti.u32(0x00FF00FF)
    x = (x | (x << 4)) & ti.u32(0x0F0F0F0F)
    x = (x | (x << 2)) & ti.u32(0x33333333)
    x = (x | (x << 1)) & ti.u32(0x55555555)
    return x

@ti.func
def morton_encode_3d(x:ti.math.vec3):
    # 2d morton code
    # x: the position of the particle
    # return: the morton code of the particle
    x_0:ti.u32 = min(max(x[0] * 1024, 0), 1023)
    x_1:ti.u32 = min(max(x[1] * 1024, 0), 1023)
    x_2:ti.u32 = min(max(x[2] * 1024, 0), 1023)
    x_0 = expand_bits_3d(x_0)
    x_1 = expand_bits_3d(x_1)
    x_2 = expand_bits_3d(x_2)
    return x_0 | (x_1 << 1) | (x_2 << 2)

@ti.func
def morton_encode_2d(x:ti.math.vec2):
    # 2d morton code
    # x: the position of the particle
    # return: the morton code of the particle
    x_0:ti.u32 = min(max(x[0] * 1024, 0), 1023)
    x_1:ti.u32 = min(max(x[1] * 1024, 0), 1023)
    x_0 = expand_bits_2d(x_0)
    x_1 = expand_bits_2d(x_1)
    return x_0 | (x_1 << 1)

@ti.func
def morton_decode_3d(morton_code: ti.uint32) -> ti.math.vec3:
    # 3d morton code
    # morton_code: the morton code of the particle
    # return: the position of the particle
    x_0 = morton_code & 0x000003ff
    x_1 = (morton_code & 0x000ffc00) >> 10
    x_2 = (morton_code & 0x3ff00000) >> 20
    x_0 = (x_0 | (x_0 << 16)) & 0x030000ff
    x_0 = (x_0 | (x_0 << 8)) & 0x0300f00f
    x_0 = (x_0 | (x_0 << 4)) & 0x030c30c3
    x_0 = (x_0 | (x_0 << 2)) & 0x09249249
    x_1 = (x_1 | (x_1 << 16)) & 0x030000ff
    x_1 = (x_1 | (x_1 << 8)) & 0x0300f00f
    x_1 = (x_1 | (x_1 << 4)) & 0x030c30c3
    x_1 = (x_1 | (x_1 << 2)) & 0x09249249
    x_2 = (x_2 | (x_2 << 16)) & 0x030000ff
    x_2 = (x_2 | (x_2 << 8)) & 0x0300f00f
    x_2 = (x_2 | (x_2 << 4)) & 0x030c30c3
    x_2 = (x_2 | (x_2 << 2)) & 0x09249249
    return ti.Vector([x_0, x_1, x_2]) / 1024.0

@ti.func
#32bit morton code to 2d position
def morton_decode_2d(morton_code: ti.uint32) -> ti.math.vec2:
    # 2d morton code
    # morton_code: the morton code of the particle
    # return: the position of the particle
    x = morton_code & ti.u32(0x55555555)
    y = (morton_code >> 1) & ti.u32(0x55555555)
    x = (x | (x >> 1)) & ti.u32(0x33333333)
    y = (y | (y >> 1)) & ti.u32(0x33333333)
    x = (x | (x >> 2)) & ti.u32(0x0f0f0f0f)
    y = (y | (y >> 2)) & ti.u32(0x0f0f0f0f)
    x = (x | (x >> 4)) & ti.u32(0x00ff00ff)
    y = (y | (y >> 4)) & ti.u32(0x00ff00ff)
    x = (x | (x >> 8)) & ti.u32(0x0000ffff)
    y = (y | (y >> 8)) & ti.u32(0x0000ffff)
    return ti.Vector([x, y]) / 1024.0

@ti.func
def encode_2d(x:ti.template(),res:ti.template()):
    for i in x:
        res[i] = morton_encode_2d(x[i])