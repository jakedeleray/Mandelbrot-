import numpy as np
import matplotlib.pyplot as plt

#Parameters

width = 800 
height = 600 
max_iter = 100

x_min, x_max = -2.0, 1.0
y_min, y_max = -1.2, 1.2

Output_file = "mandelbrot.ppm"

def mandelbrot():
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    iters = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)  

    for i in range(1, max_iter + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = (Z.real**2 + Z.imag**2) > 4

        newly_escaped = escaped & mask
        iters[newly_escaped] = i

        mask[newly_escaped] = False

        if not mask.any():
            break

    iters[iters == 0] = max_iter
    return iters



def iterations_to_rgb(iters):
    gray = (255 * (1 - iters / max_iter)).astype(np.uint8)
    gray[iters == max_iter] = 0
    return np.stack([gray, gray, gray], axis = -1)


def write_ppm(filename, rgb):
    height, width, _ = rgb.shape
    with open(filename, "w") as f:
        f.write("P3\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")

        for row in rgb:
            f.write(" ".join(map(str, row.flatten()))+ "\n")


#run stuff
iters = mandelbrot()
rgb = iterations_to_rgb(iters)
write_ppm(Output_file, rgb)

plt.imshow(rgb)
plt.axis("off")
plt.show()

print(f"Wrote {Output_file}")