import taichi as ti
import taichi.math as tm
ti.init(arch=ti.vulkan)

w, h = 1000, 1000
res = (w, h)

pixels = ti.Vector.field(3, float, shape=res)
window = ti.ui.Window("fractal", res=res)
canvas = window.get_canvas()

vec3 = tm.vec3

@ti.kernel
def render(t: float):
    for i, j in pixels:
        col = vec3(0)
        zoom = 0.0
        for k in range(1, 100):
            p = zoom * vec3(i - 0.5 * w, j - 0.5 * h, h) / h
            p.z -= 1.0
            p = tm.rot_by_axis(tm.normalize(vec3(1, 3, 3)), t * 0.2)[0 : 3, 0: 3] @ p
            s = 3.0
            for _ in range(6):
                e = 1.0 / tm.min(1.0, tm.dot(p, p))
                s *= e
                p = abs(p) * e - 1.5
            zoom += (d:= tm.length(p.xy) / s)
            col += (tm.cos(zoom * 6.28 + vec3(0, 22, 21)) * 0.24 + 0.56) \
                * float(d < 0.001) / k
        pixels[i, j] = col

import time
t0 = time.perf_counter()
while window.running:
    render(time.perf_counter() - t0)
    canvas.set_image(pixels)
    window.show()