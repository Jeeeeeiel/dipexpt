# /usr/bin/python3
# __*__ coding: utf-8 __*__

import numpy as np
import random
import math
from dipexpt1 import BMP
from dipexpt2 import save
from dipexpt3 import Integral_Histogram


def binarization_using_otsu(data):
    i_histogram = Integral_Histogram(data)[-1, -1]
    intensity_probability = i_histogram / np.sum(i_histogram)
    Sigma2 = np.zeros(i_histogram.shape[0])  # means between-class variance
    # print(Sigma2.shape)
    mG = np.sum(range(256) * intensity_probability)  # mG
    P1k = 0  # P1(k)
    mk = 0
    for i in range(Sigma2.shape[0]):
        P1k += intensity_probability[i]
        if P1k == 0:
            continue
        if P1k == 1:
            break
        mk += i * intensity_probability[i]
        Sigma2[i] = (mG * P1k - mk) ** 2 / P1k / (1 - P1k)

    # print(Sigma2)
    # print(np.argmax(Sigma2))
    # print(np.max(Sigma2))
    k = np.argmax(Sigma2)
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = (data[:, :, 0] > k) * 255
    return data


def binarization_using_half_tone(data):
    x = np.zeros((data.shape[0] + 2, data.shape[1] + 4))  # pad, left:2, right:2,bottom:2
    x[:-2, 2: -2] = data[:, :, 0]
    y = np.zeros(data[:, :, 0].shape)
    TH = 0.5 * 255
    K = np.array([[0, 0, 0, 7, 5], [3, 5, 7, 5, 3], [1, 3, 5, 3, 1]]) / 48
    m = K.shape[0]
    n = K.shape[1]
    for i in range(data[:, :, 0].shape[0]):
        for j in range(data[:, :, 0].shape[1]):
            y[i, j] = 255 * (x[i, j + 2] > TH)
            e = x[i, j + 2] - y[i, j]
            x[i: i + m, j: j+n] = x[i: i + m, j: j+n] + K * e
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = y
    return data


def salt_and_pepper_noise(data, SNR=0.2):  # SNR from 0.0 to 1.0
    height = data.shape[0]
    width = data.shape[1]
    size = height * width
    samples = random.sample(range(size), math.ceil(size * SNR))
    for s in samples:
        data[s // height, s % height, 0] = (random.random() >= 0.5) * 255
    data[:, :, 1] = data[:, :, 2] = data[:, :, 0]
    return data


def gaussian_noise(data, mu=0, sigma=20):
    tmpdata = np.zeros(data.shape[0: 2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            tmpdata[i, j] = data[i, j, 0] + random.gauss(mu, sigma)
            if tmpdata[i, j] > 255:
                tmpdata[i, j] = 255
            if tmpdata[i, j] < 0:
                tmpdata[i, j] = 0
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = tmpdata
    return data


def median_filter(data):
    pass


def main():
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Color128/LENA.BMP')
    bmp.change_to_gray()
    data = binarization_using_otsu(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/binary.bmp')
    data = binarization_using_half_tone(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/half-tone.bmp')
    data = salt_and_pepper_noise(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/salt_and_pepper_noise.bmp')

    data = gaussian_noise(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/gaussian_noise.bmp')


if __name__ == '__main__':
    main()
