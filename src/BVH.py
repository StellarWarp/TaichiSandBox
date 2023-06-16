import taichi as ti
import morton
from sort import radix_sort

# 最长公共前缀
@ti.func
def longest_common_prefix(a: int, b: int):
    # longest common prefix
    # a: the first integer
    # b: the second integer
    # return: the longest common prefix of a and b
    return ti.count_trailing_zeros(ti.bit(a ^ b))



@ti.dataclass
class BoundingBox:
    # bounding box
    # min: the minimum coordinate of the bounding box
    # max: the maximum coordinate of the bounding box
    min: ti.math.vec2
    max: ti.math.vec2


@ti.dataclass
class BVHNode:
    # BVH node
    # left: the left child
    # right: the right child
    # parent: the parent node
    # leaf: whether the node is a leaf node
    # depth: the depth of the node
    # morton_code: the morton code of the node
    # bounding_box: the bounding box of the node
    left: ti.uint32
    right: ti.uint32
    parent: ti.uint32
    depth: int
    morton_code: int
    bounding_box: BoundingBox


@ti.dataclass
class BVHLeaf:
    # BVH leaf node
    # index: the index of the particle
    # bounding_box: the bounding box of the particle
    index: int
    morton_code: int
    bounding_box: BoundingBox


@ti.data_oriented
class BVH:
    def __init__(self, n_obj: int):
        # set n_obj to nearest power of 2
        self.n_obj = 1 << (n_obj - 1).bit_length()
        self.leaves = BVHLeaf.field(shape=n_obj)
        self.nodes = BVHNode.field(shape=n_obj * 2 - 1)
        
    @ti.kernel
    def construct_0(self, x: ti.template()):
        morton.encode_2d(x, self.morton_code_space)
    @ti.kernel
    def construct_1(self, x: ti.template()):
        for i in x:
            self.leaves[i] = BVHLeaf(i, BoundingBox(x[i], x[i]))
        

    def construct(self, x: ti.template()):
        index_map = ti.field(dtype=ti.u32, shape=x.shape[0])
        radix_sort(x, index_map)
        self.construct_1(x, index_map)





ti.init(arch=ti.gpu)

gui = ti.GUI("Taichi ", res=512, background_color=0x112F41)


n_particles = 1024
x = ti.Vector.field(2, dtype=float, shape=n_particles)
v = ti.Vector.field(2, dtype=float, shape=n_particles)
temp = ti.Vector.field(2, dtype=float, shape=n_particles)
morton_code_space = ti.field(dtype=ti.u32,shape=n_particles)
temp_field = ti.field(dtype=ti.u32,shape=n_particles)
index_map = ti.field(dtype=ti.u32,shape=n_particles)

line_begin = ti.Vector.field(2, dtype=float, shape=n_particles)
line_end = ti.Vector.field(2, dtype=float, shape=n_particles)
# random initial position
@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]
        morton_code_space[i] = morton.morton_encode_2d(x[i])


@ti.kernel
def init_line():
    for i in x:
        temp[index_map[i]] = x[i]
        temp_field[index_map[i]] = morton_code_space[i]
    for i in x:
        x[i] = temp[i]
        morton_code_space[i] = temp_field[i]
    for i in range(n_particles):
        line_begin[i] = x[i]
        line_end[i] = x[min(i+1,n_particles-1)]
    

def main():
    bvh:BVH = BVH(n_particles)
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == 'r':
                init()
        
        #draw line
        gui.lines(begin=line_begin.to_numpy(), end=line_end.to_numpy(), color=0x66ccff, radius=1.5)
        # gui.circles(x.to_numpy(), radius=1.5, color=0x66ccff)
        gui.show()



if __name__ == '__main__':
    init()
    radix_sort(morton_code_space,index_map)
    init_line()
    main()
