
from numba import cuda
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from matplotlib.pylab import show
from timeit import default_timer as timer


def draw_image(mat, cmap='inferno', powern=0.5, dpi=72):
    ## Value normalization
    # Apply power normalization, because number of iteration is
    # distributed according to a power law (fewer pixels have
    # higher iteration number)
    mat = np.power(mat, powern)

    # Colormap: set the color the black for values under vmin (inner points of
    # the set), vmin will be set in the imshow function
    new_cmap = copy.copy(cm.get_cmap(cmap))
    new_cmap.set_under('black')

    ## Plotting image

    # Figure size
    plt.figure(figsize=(mat.shape[0] / dpi, mat.shape[1] / dpi))

    # Plotting mat with cmap
    # vmin=1 because smooth iteration count is always > 1
    # We need to transpose mat because images use row-major
    # ordering (C convention)
    # origin='lower' because mat[0,0] is the lower left pixel
    plt.imshow(mat.T, cmap=new_cmap, vmin=1, origin='lower')

    # Remove axis and margins
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    show()

@cuda.jit
def mandelbrot(x, y, max_iters):
    c = complex(x, y)
    z = 0
    n = 0

    while abs(z) <= 4 and n < max_iters:
        z = z * z + c
        n += 1
    return n

@cuda.jit
def create_fractal(image, iters=100,x_min=-2.6, x_max=1.85, y_min=-1.25, y_max=1.25):
    # получение индекса текущего блока
    x = cuda.blockIdx.x
    # получение индекса текущего потока
    y = cuda.threadIdx.x
    # сопоставление пикселей и комплексных чисел
    real = x_min + x / image.shape[0] * (x_max - x_min)
    imag = y_min + y / image.shape[1] * (y_max - y_min)
    # нахождение цвета элемента
    color = mandelbrot(real, imag, iters)
    image[x,y] = color

# xmin, xmax = -1.7687782, -1.7687794
# ymin, ymax = -0.0017384, -0.0017394
xmin, xmax = -2.0,1.0
ymin, ymax = -.50,.50

xpixels = 75
ypixels = round(xpixels / (xmax-xmin) * (ymax-ymin))
image = np.zeros((xpixels, ypixels))
if (ypixels > 1024):
    ypixels = 1024
iters = 50
# maxiter = 50


# Running and plotting result
start = timer()
create_fractal[xpixels, ypixels](image, iters,xmin, xmax ,ymin, ymax)
end = timer()
print("Exec time on GPU: %f "%(end-start))
draw_image(image)
