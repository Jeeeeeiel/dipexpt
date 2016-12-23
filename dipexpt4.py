# /usr/bin/python3
# __*__ coding: utf-8 __*__

import numpy as np
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
    pass




def main():
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Color128/LENA.BMP')
    bmp.change_to_gray()
    data = binarization_using_otsu(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/binary.bmp')
    data = binarization_using_half_tone(bmp.data.copy())
    save(bmp, data, '/Users/Jeiel/Desktop/half-tone.bmp')

if __name__ == '__main__':
    main()