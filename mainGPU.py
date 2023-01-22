import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from matplotlib.pylab import show
from timeit import default_timer as timer
from numba import cuda
def draw_image(image):
    image = np.power(image, 0.8)
    # установка желаемой цветовой карты
    cmap = 'hot' # gist_stern, inferno, gist_ncar, hot
    new_cmap = copy.copy(cm.get_cmap(cmap))
    new_cmap.set_under('black') # установка цвета фона
    # построение изображения
    # задание размеров выводимого изображения
    dpi = 500
    plt.figure(figsize=(image.shape[0] / dpi, image.shape[1] / dpi))
    # построение изображения
    plt.imshow(image.T, cmap=new_cmap, vmin=1, origin='lower')
    # удаление осей и полей
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    # отображение картинки
    show()
@cuda.jit
def mandelbrot(x,y, max_iters):
    c = complex(x,y)
    z = 0
    n = 0

    while abs(z) <= 1.5 and n < max_iters:
        z = z * z + c
        n += 1
    return n
@cuda.jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    # ширина картинки
    width = image.shape[0]
    # высота картинки
    height = image.shape[1]
    # вычисление ширины 1 пикселя
    pixel_size_x = (max_x - min_x) / width
    # вычисление высоты 1 пикселя
    pixel_size_y = (max_y - min_y) / height
    # передача абсолютной позиции текущего потока во всей сетке блоков
    x, y = cuda.grid(2)
    # условие обработки пикселей
    if x < width and y < height:
        # вычисление действительной части числа
        real = min_x + x * pixel_size_x
        # вычисление мнимой части числа
        imag = min_y + y * pixel_size_y
        # нахождение цвета элемента
        color = mandelbrot(real, imag, iters)
        image[x,y] = color
#### for first set
h = 5000
w = 7500
#### for second set
# h = 2500
# w = 3500
#### for second set
# h = 1017
# w = 1220

image = np.zeros((w,h))

pixels = h * w

nthreads = 32 # потоков на блок

nblocksy = int(np.ceil((h)/nthreads))
nblocksx = int(np.ceil((w)/nthreads))
################################## first set
start = timer()
# (nblocksx,nblocksy) блоков на грид
# (32,32) потоков на блок
create_fractal[(nblocksx,nblocksy),(nthreads,nthreads)](-2.0,1.0,-1.0,1.0, image, 50) # вызов ядра cuda
end = timer()
################################ second set
# start = timer()
# # (nblocksx,nblocksy) блоков на грид
# # (32,32) потоков на блок
# create_fractal[(nblocksx,nblocksy),(32,32)](-1.7687782, -1.7687794,-0.0017384, -0.0017394, image,10000) # вызов ядра cuda
# end = timer()

print("Exec time on GPU: %f "%(end-start))
draw_image(image)

