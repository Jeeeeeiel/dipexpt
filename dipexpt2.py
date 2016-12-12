# /usr/bin/python3
import numpy as np
from dipexpt1 import BMP
from math import pi
from math import sqrt
from scipy.fftpack import dct
from scipy.fftpack import idct
import struct

StdQuanTable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                        [12, 12, 14, 19, 26, 58, 60, 55],
                        [14, 13, 16, 24, 40, 57, 69, 56],
                        [14, 17, 22, 29, 51, 87, 80, 62],
                        [18, 22, 37, 56, 68, 109, 103, 77],
                        [24, 35, 55, 64, 81, 104, 113, 92],
                        [49, 64, 78, 87, 103, 121, 120, 101],
                        [72, 92, 95, 98, 112, 100, 103, 99]])


def DCT(data):  # deal with square data, x==y
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

    # for check
    # tmpdata = data[:, :, 0]
    # tmpdata = dct(tmpdata.T, norm='ortho')
    # tmpdata = dct(tmpdata.T, norm='ortho')
    # tmpdata = idct(tmpdata.T, norm='ortho')
    # tmpdata = idct(tmpdata.T, norm='ortho')

    data = np.zeros(data.shape, dtype=tmpdata.dtype)  # change dtype
    data[:, :, 1] = data[:, :, 2] = data[:, :, 0] = tmpdata
    return data


def IDCT(data):  # deal with square data, x==y
    # for check
    # data = idct(data[:, :, 0].T, norm='ortho')
    # data = idct(data.T, norm='ortho')
    # tmpdata = np.ones((*data.shape, 3), dtype=data.dtype)
    # tmpdata[:, :, 0] = tmpdata[:, :, 1] = tmpdata[:, :, 2] = data
    # data = np.array(tmpdata, dtype=np.uint8)
    # return data

    trans_mat = np.ones(data.shape[0:2])  # left *
    N = data.shape[0]  # height

    # init convert matrix
    E = np.ones((1, N))
    E[0, 0] = sqrt(2) / 2  # E = [[sqrt(2)/2, 1, 1, 1, ..., 1]]
    for row in range(N):
        trans_mat[row] = E * np.cos((2 * trans_mat[row] * row + 1) * range(N) * pi / 2 / N)
    # print(trans_mat[1])

    # trans y
    tmpdata = np.dot(trans_mat, data[:, :, 0].T)  # transform in b channel

    # trans x, y==x
    tmpdata = np.dot(trans_mat, tmpdata.T)
    tmpdata = tmpdata * 2 / N
    data = np.zeros(data.shape, dtype=tmpdata.dtype)  # change dtype
    data[:, :, 1] = data[:, :, 2] = data[:, :, 0] = tmpdata
    return data


def Quantize(data):
    tmpdata = np.around(data[:, :, 0] / StdQuanTable)
    # data = np.zeros(data.shape, dtype=np.int)
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = tmpdata
    return data


def IQuantize(data):
    tmpdata = data[:, :, 0] * StdQuanTable
    data = np.zeros(data.shape, dtype=tmpdata.dtype)
    data[:, :, 0] = data[:, :, 1] = data[:, :, 2] = tmpdata
    return data


def DFT(data):  # deal with square data, x==y
    trans_mat = np.ones(data.shape[0:2])  # left *
    N = data.shape[0]  # height
    # init convert matrix
    for row in range(N):
        trans_mat[row] = row * trans_mat[row] * range(N)
    # print(trans_mat[0])
    trans_mat = np.exp(-2j * pi / N * trans_mat)
    # print(trans_mat[1])

    tmpdata = np.dot(trans_mat, data[:, :, 0])  # transform in b channel
    # for check
    # tmpdata = np.fft.fft(data[:, :, 0], axis=0)
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
    bmp.change_to_gray()  # assign the values in b channel to g r
    # whole image
    # data = np.array(DFT(bmp.data.copy()), dtype=np.uint8)
    # save(bmp, data, '/Users/Jeiel/Desktop/DFT_whole_image_y.bmp')
    # data = np.array(DCT(bmp.data.copy()), dtype=np.uint8)
    # save(bmp, data, '/Users/Jeiel/Desktop/DCT_whole_image_x_y.bmp')
    # data = np.array(IDCT(DCT(bmp.data.copy())), dtype=np.uint8)
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

    #         # subdata = np.array(DCT(subdata), dtype=np.uint8)  # DCT
    #         # subdata = np.array(IDCT(DCT(subdata)), dtype=np.uint8)  # IDCT

    #         # for check
    #         # subdata = idct(subdata[:, :, 0].T, norm='ortho')
    #         # subdata = idct(subdata.T, norm='ortho')
    #         # tmpdata = np.ones((*subdata.shape, 3), dtype=subdata.dtype)
    #         # tmpdata[:, :, 0] = tmpdata[:, :, 1] = tmpdata[:, :, 2] = subdata
    #         # subdata = tmpdata

    #         data[i * 8: i * 8 + 8, j * 8: j * 8 + 8] = subdata
    # save(bmp, data, '/Users/Jeiel/Desktop/DCT_8*8_x_y.bmp')

    # DCT: 8 * 8 block, use LEFT TOP coefficient
    # data = bmp.data.copy()
    # for i in range((data.shape[0] + 7) // 8):  # height
    #     for j in range((data.shape[1] + 7) // 8):  # width
    #         subdata = data[i * 8: i * 8 + 8, j * 8: j * 8 + 8]
    #         # print(DCT(subdata)[:, :, 0])
    #         subdata = DCT(subdata)
    #         left = 3
    #         subdata[left:, :, :] = 0
    #         subdata[:, left:, :] = 0
    #         # print(subdata[:, :, 0])
    #         subdata = np.array(IDCT(subdata), dtype=np.uint8)

    #         data[i * 8: i * 8 + 8, j * 8: j * 8 + 8] = subdata
    # save(bmp, data, '/Users/Jeiel/Desktop/LT_DCT_8*8_x_y.bmp')

    # Quantize DCT: 8 * 8 block
    # data = bmp.data.copy()
    # for i in range((data.shape[0] + 7) // 8):  # height
    #     for j in range((data.shape[1] + 7) // 8):  # width
    #         subdata = data[i * 8: i * 8 + 8, j * 8: j * 8 + 8]
    #         # print(IQuantize(Quantize(DCT(subdata)))[:, :, 0])

    #         # subdata = np.array((Quantize(DCT(subdata))), dtype=np.uint8)  # Quantize<-DCT
    #         # subdata = np.array(IQuantize(Quantize(DCT(subdata))), dtype=np.uint8)  # IQuantize<-Quantize<-DCT
    #         # subdata = np.array(IDCT(IQuantize(Quantize(DCT(subdata)))), dtype=np.uint8) # IDCT<-IQuantize<-Quantize<-DCT

    #         data[i * 8: i * 8 + 8, j * 8: j * 8 + 8] = subdata
    # save(bmp, data, '/Users/Jeiel/Desktop/Quantize_DCT_8*8_x_y.bmp')
