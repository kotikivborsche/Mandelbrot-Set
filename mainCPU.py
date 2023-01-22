import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pylab import show
from timeit import default_timer as timer

def draw_image(image):
    # установка желаемой цветовой карты
    cmap = 'hot' # gist_stern, inferno, gist_ncar, inferno, hot
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

def mandelbrot(x,y, max_iters):
    # создание комплексного числа, обнуление счетчика
    c = complex(x,y)
    z = 0
    n = 0
    # получение числа итераций для элемента множества Манделброта
    while abs(z) <= 4 and n < max_iters:
        z = z * z + c
        n += 1
    return n

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    # ширина картинки
    width = image.shape[0]
    # высота картинки
    height = image.shape[1]
    # вычисление ширины 1 пикселя
    pixel_size_x = (max_x - min_x) / width
    # вычисление высоты 1 пикселя
    pixel_size_y = (max_y - min_y) / height
    # перебор всех пикселей картинки
    for x in range(width):
        # вычисление действительной части числа
        real = min_x + x * pixel_size_x
        for y in range(height):
            # вычисление мнимой части числа
            imag = min_y + y * pixel_size_y
            # нахождение цвета элемента
            color = mandelbrot(real, imag, iters)
            image[x,y] = color
# объявление и заполнение 0 матрицы
image = np.zeros((75,50), dtype = np.uint8)
# image = np.zeros((1220,1017), dtype = np.uint8)
# начало отсчета времени вычислений
start = timer()
# вызов основной функции алгоритма
create_fractal(-2.0,1.0,-1.0,1.0, image, 50)
# конец отсчета времени вычислений
end = timer()

# s = timer()
# create_fractal(-1.7687782, -1.7687794,-0.0017384, -0.0017394, image,10000)
# e = timer()
# вывод времени работы алгоритма
print("Exec time on CPU: %f "%(end-start))
# отрисовка фрактальной картины
draw_image(image)
