# /usr/bin/python3
# __*__ coding: utf-8 __*__
import logging
import struct
import numpy as np
from math import acos
from math import sqrt

logging.basicConfig(level=logging.DEBUG)
np.seterr(over='ignore')


class BMP(object):
    def __init__(self, filename):
        self.filename = filename
        self.bitmapfileheader = {}
        self.bitmapinfo = {}
        self.bitMapInfoHeader = {}
        self.RGBQUADS = None
        self.pixels = []
        self.isBMP = False
        self.readbmp()

    def readbmp(self):
        with open(self.filename, 'rb') as file:
            bitmapfileheader = file.read(14)
            self.bitmapfileheader = struct.unpack('<ccIHHI', bitmapfileheader)
            if self.bitmapfileheader[0:2] != (b'B', b'M'):
                print('Not a .bmp file!')
                return
            self.isBMP = True
            logging.debug(self.bitmapfileheader)
            bitmapinfoheader = file.read(40)
            self.bitMapInfoHeader = struct.unpack('<IIIHHIIIIII', bitmapinfoheader)
            logging.debug(self.bitMapInfoHeader)

            if self.bitmapfileheader[5] > 54:
                self.RGBQUADS = file.read(self.bitmapfileheader[5] - 54)

            rowsize = (self.bitMapInfoHeader[1] * self.bitMapInfoHeader[4] + 31) // 32 * 4  # unit: byte
            skip = rowsize * 8 - self.bitMapInfoHeader[1] * self.bitMapInfoHeader[4]  # unit: byte
            # logging.debug(rowsize)
            # logging.debug(skip)

            if self.bitMapInfoHeader[4] == 1:
                self.data = np.zeros((self.bitMapInfoHeader[2], self.bitMapInfoHeader[1]), dtype=np.uint8)  # height, width
                for i in range(self.bitMapInfoHeader[2]):
                    row = file.read(rowsize)
                    for j in range(self.bitMapInfoHeader[1]):
                        self.data[i, j] = (row[j // 8] & (0x01 << (7 - j % 8))) >> (7 - j % 8)  # get the responding bit than shift to right

            elif self.bitMapInfoHeader[4] == 4:
                pass
            elif self.bitMapInfoHeader[4] == 8:
                pass
            elif self.bitMapInfoHeader[4] == 24:
                self.data = np.zeros((self.bitMapInfoHeader[2], self.bitMapInfoHeader[1], 3), dtype=np.uint8)  # height, width
                for i in range(self.bitMapInfoHeader[2]):
                    row = file.read(rowsize)
                    for j in range(self.bitMapInfoHeader[1]):
                        self.data[i, j, 0] = row[j * 3]  # b
                        self.data[i, j, 1] = row[j * 3 + 1]  # g
                        self.data[i, j, 2] = row[j * 3 + 2]  # r
            elif self.bitMapInfoHeader[4] == 32:
                pass

    def saveNonRGB(self):
        with open('/Users/Jeiel/Desktop/notRBG' + str(self.bitMapInfoHeader[4]) + '.bmp', 'wb') as file:
            file.write(struct.pack('<ccIHHI', *self.bitmapfileheader))
            file.write(struct.pack('<IIIHHIIIIII', *self.bitMapInfoHeader))
            file.write(self.RGBQUADS)
            print('*********')
            for i in range(self.bitMapInfoHeader[2]):  # height
                rowlenth = 0  # unit: byte
                for j in range((self.bitMapInfoHeader[1] + 7) // 8):  # total byte in every row before pad
                    B = 0x00
                    for b in self.data[i, j * 8:j * 8 + 8]:  # semble bits into byte
                        B = (B + b) << 1
                    B = B >> 1
                    B = B << (8 - len(self.data[i, j:j + 8]))  # the number of bits left less than 8
                    file.write(bytes([B]))
                    rowlenth += 1
                if rowlenth % 4 != 0:  # row needs to padded
                    # print(rowlenth)
                    file.write(bytes([0x00] * (4 - rowlenth % 4)))

    def saveRGB(self, R=False, G=False, B=False):
        tmpdata = np.zeros((self.bitMapInfoHeader[2], self.bitMapInfoHeader[1], 3), dtype=np.uint8)
        tmpdata = self.data * np.array([B, G, R])
        with open('/Users/Jeiel/Desktop/RGB' + str(R) + str(G) + str(B) + '.bmp', 'wb') as file:
            file.write(struct.pack('<ccIHHI', *self.bitmapfileheader))
            file.write(struct.pack('<IIIHHIIIIII', *self.bitMapInfoHeader))
            for B in tmpdata:
                file.write(B)

    def saveYIQorXYZ(self, YIQ=[False, False, False], XYZ=[False, False, False]):
        if YIQ[0] or YIQ[1] or YIQ[2]:
            s = 'YIQ'
            trans_matrix = np.array([[0.299, 0.587, 0.114],
                                    [0.596, -0.274, -0.322],
                                    [0.211, -0.523, 0.312]]) * np.array([[YIQ[0], YIQ[1], YIQ[2]]]).T
            # reverse_trans_matrix = np.array(np.mat(np.array([[0.299, 0.587, 0.114],
            #                                                 [0.596, -0.274, -0.322],
            #                                                 [0.211, -0.523, 0.312]])).I)
        elif XYZ[0] or XYZ[1] or XYZ[2]:
            s = 'XYZ'
            trans_matrix = np.array([[0.490, 0.310, 0.200],
                                    [0.177, 0.813, 0.011],
                                    [0.000, 0.010, 0.990]]) * np.array([[XYZ[0], XYZ[1], XYZ[2]]]).T
            # reverse_trans_matrix = np.array(np.mat(np.array([[0.490, 0.310, 0.200],
            #                                              [0.177, 0.813, 0.011],
            #                                              [0.000, 0.010, 0.990]])).I)
        tmpdata = np.zeros((self.bitMapInfoHeader[2], self.bitMapInfoHeader[1], 3), dtype=np.uint8)
        for i in range(self.bitMapInfoHeader[2]):
            for j in range(self.bitMapInfoHeader[1]):
                tmpdata[i, j] = trans_matrix.dot(self.data[i, j, ::-1].T)
                # tmpdata[i, j] = reverse_trans_matrix.dot(tmpdata[i, j].T)[::-1]

        # img = cv.merge((tmpdata[::-1, :, 0], tmpdata[::-1, :, 1], tmpdata[::-1, :, 1]))
        # cv.imwrite('/Users/Jeiel/Desktop/' + s + str(YIQ[0] or XYZ[0]) + str(YIQ[1] or XYZ[1]) + str(YIQ[2] or XYZ[2]) + '.jpg', img)

        with open('/Users/Jeiel/Desktop/' + s + str(YIQ[0] or XYZ[0]) + str(YIQ[1] or XYZ[1]) + str(YIQ[2] or XYZ[2]) + '.bmp', 'wb') as file:
            file.write(struct.pack('<ccIHHI', *self.bitmapfileheader))
            file.write(struct.pack('<IIIHHIIIIII', *self.bitMapInfoHeader))
            for B in tmpdata:
                file.write(B)

    def saveHSI(self, H=False, S=False, I=False):
        tmpdata = np.zeros((self.bitMapInfoHeader[2], self.bitMapInfoHeader[1], 3), dtype=np.uint8)
        for i in range(self.bitMapInfoHeader[2]):
            for j in range(self.bitMapInfoHeader[1]):
                tmpdata[i, j, 2] = I * sum(self.data[i, j]) / 3  # I
                tmpdata[i, j, 1] = S * (1 - 3 / sum(self.data[i, j]) * min(self.data[i, j]))  # S
                tmpdata[i, j, 0] = H * acos((self.data[i, j, 2] - sum(self.data[i, j, 0:2]) / 2)
                                        / sqrt((self.data[i, j, 2] ** 2 + (self.data[i, j, 2] - self.data[i, j, 0]) * (self.data[i, j, 1] - self.data[i, j, 0]))))  # H
        with open('/Users/Jeiel/Desktop/HSI' + str(H) + str(S) + str(I) + '.bmp', 'wb') as file:
            file.write(struct.pack('<ccIHHI', *self.bitmapfileheader))
            file.write(struct.pack('<IIIHHIIIIII', *self.bitMapInfoHeader))
            for B in tmpdata:
                file.write(B)


if __name__ == '__main__':
    binary image
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Bin64/TUANJIE.BMP')
    bmp.saveNonRGB()
    bmp = BMP('/Users/Jeiel/Dropbox/数字图像处理/实验/实验一素材/Color128/LENA.BMP')

    # color image
    # if bmp.isBMP and bmp.bitMapInfoHeader[4] == 24:
    #     bmp.saveRGB(R=True, G=True, B=True)
    #     bmp.saveRGB(R=True)
    #     bmp.saveRGB(G=True)
    #     bmp.saveRGB(B=True)
    #     bmp.saveYIQorXYZ(YIQ=[True, False, False])
    #     bmp.saveYIQorXYZ(YIQ=[False, True, False])
    #     bmp.saveYIQorXYZ(YIQ=[False, False, True])
    #     bmp.saveYIQorXYZ(XYZ=[True, False, False])
    #     bmp.saveYIQorXYZ(XYZ=[False, True, False])
    #     bmp.saveYIQorXYZ(XYZ=[False, False, True])
    #     bmp.saveHSI(H=True)
    #     bmp.saveHSI(S=True)
    #     bmp.saveHSI(I=True)

