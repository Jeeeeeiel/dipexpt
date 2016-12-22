# /usr/bin/python3
# __*__ coding: utf-8 __*__

from dipexpt1 import BMP
from dipexpt2 import save
import numpy as np


def sobel(data):  # sobel 3*3

    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = Sx.T
    paddata = np.pad(data[:, :, 0], (1, 1), 'edge')
    paddata[1: -1, 1: -1] = data[:, :, 0]
    tmpdata = np.array(data[:, :, 0], dtype=np.float)
    for i in range(tmpdata.shape[0]):
        for j in range(tmpdata.shape[1]):
            tmpdata[i, j] = np.sqrt((np.sum(paddata[i: i + 3, j: j + 3] * Sx)) ** 2 + (np.sum(paddata[i: i + 3, j: j + 3] * Sy)) ** 2)

    tmpdata = tmpdata / np.max(tmpdata) * 255
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = tmpdata
    return data


def Integral_Image(data):
    i_image = np.zeros(data.shape[0:2])
    for i in range(i_image.shape[0]):
        for j in range(i_image.shape[1]):
            i_image[i, j] = i_image[i - 1, j - 1] + np.sum(data[i, 0: j, 0]) + np.sum(data[0: i, j, 0])
    return i_image


def Integral_Histogram(data):
    gray_scale = np.array(range(256))
    i_histogram = np.zeros((*data.shape[0:2], 256))
    for i in range(i_histogram.shape[0]):
        for j in range(i_histogram.shape[1]):
            for gray in gray_scale:
                i_histogram[i, j, gray] = i_histogram[i - 1, j - 1, gray] + np.sum(data[i, 0: j, 0] == gray) + np.sum(data[0: i, j, 0] == gray)
    return i_histogram


def main():
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Color128/LENA.BMP')
    bmp.change_to_gray()
    # save(bmp, bmp.data, '/Users/Jeiel/Desktop/tmp.bmp')
    data = sobel(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/tmp.bmp')
    i_image = Integral_Image(bmp.data.copy())
    print(i_image)
    # i_histogram = Integral_Histogram(bmp.data.copy())

if __name__ == '__main__':
    main()
