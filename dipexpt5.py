# /usr/bin/python3
# __*__ coding: utf-8 __*__

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageEnhance
# from scipy import misc
# from scipy import ndimage
import matplotlib.pyplot as plt

def horizontal_hist(data):
    hist = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        hist[i] = np.sum(data[i] == 0)  # count black
    return hist

def vertical_hist(data):
    pass

def split_text_line(im, data):
    hist = horizontal_hist(data)
    draw = ImageDraw.Draw(im)
    min_line_height = 8  # min in sample image: 11
    line_height = 0
    text_line_record = list()
    for i in range(hist.shape[0]):
        if hist[i] < 3:  # min in sample: 4
            if line_height >= min_line_height:
                text_line_record.append((i - line_height, i - 1))
            draw.line([(0, i), (data.shape[1], i)])
            if 0 < line_height < min_line_height:
                for j in range(1, line_height + 1):
                    draw.line([(0, i - j), (data.shape[1], i - j)])
            line_height = 0
        else:
            line_height += 1
    # print(text_line_record)
    # im.show()
    return text_line_record

def split_text_column(text_line_record, data):
    min_column_width = 3
    for i in range(len(text_line_record)):
        tmpdata = data[text_line_record[i][0]: text_line_record[i][1], :]


def min_filter(data, size=3):  # block: size * size
    tmpdata = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            tmpdata[i, j] = np.min(data[(i - 1 >= 0) * (i - size // 2): i + size // 2, (j - 1 >= 0) * (j - size // 2): j + size // 2])
    return tmpdata


def max_filter(data, size=3):  # block: size * size
    tmpdata = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            tmpdata[i, j] = np.max(data[(i - 1 >= 0) * (i - size // 2): i + size // 2, (j - 1 >= 0) * (j - size // 2): j + size // 2])
    return tmpdata


def binarize(im):
    im = im.convert('L')
    pixels = im.load()
    data = np.zeros((im.height, im.width))
    for i in range(im.width):  # binarize
        for j in range(im.height):
            pixels[i, j] = 255 if pixels[i, j] > 150 else 0
            data[j, i] = pixels[i, j]
    return data


def main():
    im = Image.open('/Users/Jeiel/Dropbox/数字图像处理/实验/实验五-内容和素材/sample-24 copy.jpg')
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(2)
    data = binarize(im)
    # im = im.filter(ImageFilter.MinFilter(3))
    # im = im.filter(ImageFilter.MaxFilter(3))
    # im.show()
    # plt.figure()
    # plt.imshow(data, cmap="gray")
    # plt.show()
    text_line_record = split_text_line(im, data)
    # data = min_filter(data)
    data = max_filter(data)
    plt.figure()
    for i in range(len(text_line_record)):
        plt.subplot(len(text_line_record), 1, i + 1)
        plt.imshow(data[text_line_record[i][0]: text_line_record[i][1], :], cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
