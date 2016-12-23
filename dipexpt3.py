# /usr/bin/python3
# __*__ coding: utf-8 __*__

from dipexpt1 import BMP
from dipexpt2 import save
import numpy as np
import matplotlib.pyplot as plt


def sobel(data):  # sobel 3*3

    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = Sx.T
    paddata = np.pad(data[:, :, 0], (1, 1), 'edge')
    tmpdata = np.array(data[:, :, 0], dtype=np.float)
    for i in range(tmpdata.shape[0]):
        for j in range(tmpdata.shape[1]):
            tmpdata[i, j] = np.sqrt((np.sum(paddata[i: i + 3, j: j + 3] * Sx)) ** 2 + (np.sum(paddata[i: i + 3, j: j + 3] * Sy)) ** 2)

    tmpdata = tmpdata / np.max(tmpdata) * 255
    # scale = 50
    # cutoff = scale * np.mean(tmpdata)
    # thresh = np.sqrt(cutoff)
    # tmpdata = (tmpdata > thresh) * 255
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = tmpdata
    return data


def Integral_Image(data):
    i_image = np.zeros(data.shape[0:2])
    for i in range(i_image.shape[0]):
        for j in range(i_image.shape[1]):
            i_image[i, j] = (j > 0) * i_image[i, j - 1] + (i > 0) * i_image[i - 1, j] - (i * j > 0) * i_image[i - 1, j - 1] + data[i, j, 0]

    return i_image


def Integral_Histogram(data):
    i_histogram = np.zeros((*data.shape[0:2], 256))

    for i in range(i_histogram.shape[0]):
        for j in range(i_histogram.shape[1]):
            i_histogram[i, j] = (j > 0) * i_histogram[i, j - 1] + (i > 0) * i_histogram[i - 1, j] - (i * j > 0) * i_histogram[i - 1, j - 1]
            i_histogram[i, j][data[i, j, 0]] += 1
    return i_histogram


def count_pixels_value(i_image, lx=0, ly=0, width=0, height=0):
    if width <= 0:
        width = i_image.shape[1]
    if height <= 0:
        height = i_image.shape[0]
    data = i_image[ly + height - 1, lx + width - 1] -\
        (ly > 0) * i_image[ly - 1, lx + width - 1] -\
        (lx > 0) * i_image[ly + height - 1, lx - 1] +\
        (lx * ly > 0) * i_image[ly - 1, lx - 1]

    return data


def show_Histogram(i_histogram, lx=0, ly=0, width=0, height=0):
    if width <= 0:
        width = i_histogram.shape[1]
    if height <= 0:
        height = i_histogram.shape[0]
    data = i_histogram[ly + height - 1, lx + width - 1] -\
        (ly > 0) * i_histogram[ly - 1, lx + width - 1] -\
        (lx > 0) * i_histogram[ly + height - 1, lx - 1] +\
        (lx * ly > 0) * i_histogram[ly - 1, lx - 1]
    fig = plt.figure()
    rect = plt.bar(range(256), data)
    plt.axis([0, 255, 0, np.max(data) + 2])
    # print(data)
    plt.show()


def main():
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Color128/LENA.BMP')
    bmp.change_to_gray()
    # save(bmp, bmp.data, '/Users/Jeiel/Desktop/tmp.bmp')
    data = sobel(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/sobel.bmp')

    i_image = Integral_Image(bmp.data.copy())
    # print(count_pixels_value(i_image))
    # print(np.sum(bmp.data[:, :, 0]))  # for check
    # print(count_pixels_value(i_image, lx=10, ly=10, width=40, height=30))
    # print(np.sum(bmp.data[10: 10 + 30, 10: 10 + 40, 0]))  # for check

    i_histogram = Integral_Histogram(bmp.data.copy())
    # show_Histogram(i_histogram)
    # show_Histogram(i_histogram, lx=0, ly=0, width=30, height=30)


if __name__ == '__main__':
    main()
