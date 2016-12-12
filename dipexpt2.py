# /usr/bin/python3
import numpy as np
from dipexpt1 import BMP
from math import pi
from math import sqrt
from scipy.fftpack import dct
from scipy.fftpack import idct
import struct


def DCT(data):
    trans_mat = np.ones(data.shape[0:2])  # left *
    N = data.shape[0]  # height

    # trans y
    # init convert matrix
    for row in range(N):
        trans_mat[row] = (2 * trans_mat[row] * range(N) + 1) * row
    trans_mat = np.cos(trans_mat * pi / 2 / N)
    # print(trans_mat[1])

    tmpdata = np.dot(trans_mat, data[:, :, 0])  # transform in b channel

    # trans x,x==y
    tmpdata = tmpdata.T
    tmpdata = np.dot(trans_mat, tmpdata).T

    coefficient = np.ones(data.shape[0: 2]) * 2 / N  # height*weight

    coefficient[0, :] = coefficient[0, :] * sqrt(2) / 2
    coefficient[:, 0] = coefficient[:, 0] * sqrt(2) / 2
    tmpdata = tmpdata * coefficient

    # tmpdata = data[:, :, 0]
    # tmpdata = dct(tmpdata.T, norm='ortho')
    # tmpdata = dct(tmpdata.T, norm='ortho')
    # tmpdata = idct(tmpdata.T, norm='ortho')
    # tmpdata = idct(tmpdata.T, norm='ortho')

    data = np.zeros(data.shape, dtype=tmpdata.dtype)  # change dtype
    data[:, :, 1] = data[:, :, 2] = data[:, :, 0] = tmpdata
    return data


def IDCT(bmp):
    pass


def DFT(data):
    trans_mat = np.ones(data.shape[0:2])  # left *
    N = data.shape[0]  # height
    # init convert matrix
    for row in range(N):
        trans_mat[row] = row * trans_mat[row] * range(N)
    # print(trans_mat[0])
    trans_mat = np.exp(-2j * pi / N * trans_mat)
    # print(trans_mat[1])

    tmpdata = np.dot(trans_mat, data[:, :, 0])  # transform in b channel
    # tmpdata = np.fft.fft(data[:, :, 0], axis=0)  # check
    # tmpdata = np.fft.ifft(tmpdata, axis=0)

    data = np.zeros(data.shape, dtype=tmpdata.dtype)  # change dtype
    data[:, :, 1] = data[:, :, 2] = data[:, :, 0] = tmpdata
    return data


def save(bmp, data, filename):
    with open(filename, 'wb') as file:
        file.write(struct.pack('<ccIHHI', *bmp.bitmapfileheader))
        file.write(struct.pack('<IIIHHIIIIII', *bmp.bitMapInfoHeader))
        for B in data:
            file.write(B)


if __name__ == '__main__':
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Color128/LENA.BMP')
    bmp.change_to_YCrCb()
    # # whole image
    # data = np.array(DFT(bmp.data.copy()), dtype=np.uint8)
    # save(bmp, data, '/Users/Jeiel/Desktop/DFT_whole_image_y.bmp')
    # data = np.array(DCT(bmp.data.copy()), dtype=np.uint8)
    # save(bmp, data, '/Users/Jeiel/Desktop/DCT_whole_image_x_y.bmp')
    # data = np.array(IDCT(bmp.data.copy()), dtype=np.uint8)
    # save(bmp, data, '/Users/Jeiel/Desktop/IDCT_whole_image_x_y.bmp')

    # DFT: 8 * 8 block
    # data = bmp.data.copy()
    # for i in range((data.shape[0] + 7) // 8):  # height
    #     for j in range((data.shape[1] + 7) // 8):  # width
    #         subdata = data[i * 8: i * 8 + 8, j * 8: j * 8 + 8]
    #         subdata = np.array(DFT(subdata), dtype=np.uint8)
    #         # subdata = np.fft.ifft(subdata, axis=0)  # check
    #         data[i * 8: i * 8 + 8, j * 8: j * 8 + 8] = subdata
    # save(bmp, data, '/Users/Jeiel/Desktop/DFT_8*8_y.bmp')

    # DCT: 8 * 8 block
    # data = bmp.data.copy()
    # for i in range((data.shape[0] + 7) // 8):  # height
    #     for j in range((data.shape[1] + 7) // 8):  # width
    #         subdata = data[i * 8: i * 8 + 8, j * 8: j * 8 + 8]
    #         subdata = np.array(DCT(subdata), dtype=np.uint8)
    #         # subdata = idct(subdata[:, :, 0].T, norm='ortho')
    #         # subdata = idct(subdata.T, norm='ortho')
    #         # tmpdata = np.ones((*subdata.shape, 3), dtype=subdata.dtype)
    #         # tmpdata[:, :, 0] = tmpdata[:, :, 1] = tmpdata[:, :, 2] = subdata
    #         # subdata = tmpdata
    #         data[i * 8: i * 8 + 8, j * 8: j * 8 + 8] = subdata
    # save(bmp, data, '/Users/Jeiel/Desktop/DFT_8*8_y.bmp')
