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
import math
import os
# import pytesseract


def horizontal_hist(data):
    hist = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        hist[i] = np.sum(data[i] == 0)  # count black
    return hist


def vertical_hist(data):
    hist = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        hist[i] = np.sum(data[:, i])
    return hist


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


def split_text_column(data):  # data contains single text line
    min_column_width = data.shape[0] * 0.4
    max_column_width = data.shape[0] * 1.0
    hist = vertical_hist(data)
    text_column_record = list()
    max_value = np.max(hist)
    column_width = 0
    threshold = max_value * 0.8
    i = 0
    while i < data.shape[1]:
        if hist[i] > threshold:
            if 0 < column_width < min_column_width:
                column_width += 1
                if i > 1 and (i - 2 not in text_column_record) and (i - 1 not in text_column_record) and hist[i - 1] == max_value and hist[i] == max_value:
                    text_column_record.append(i - 2)
                    text_column_record.append(i - 1)
                    text_column_record.append(i)
                    column_width = 0
                i += 1
                continue
            if min_column_width <= column_width < max_column_width:
                if (i + 1) < (data.shape[1] - 1) and hist[i] <= np.max(hist[i + 1: i + math.ceil(max_column_width) - column_width + 1]):
                    i = i + np.argmax(hist[i + 1: i + math.ceil(max_column_width) - column_width + 1]) + 1
                # elif (i + 1) < (data.shape[1] - 1) and hist[i] == max_value:
                #     continue_appear = 0
                #     for j in range(1, len(hist[i + 1: i + math.ceil(max_column_width) - column_width + 1]) + 1):
                #         if hist[i + j] == max_value:
                #             continue_appear += 1
                #             if continue_appear > 1:
                #                 i = i + j - 1
                #                 break
                #         else:
                #             continue_appear = 0
            elif column_width > 1.3 * max_column_width:
                i = i - column_width
                i = i + math.ceil(max_column_width * 0.8) + np.argmax(hist[i + math.ceil(max_column_width * 0.8): i + math.ceil(max_column_width * 1.2)]) + 1
            text_column_record.append(i)
            column_width = 0
            i += 1
        else:
            i += 1
            column_width += 1
    # for show
    # im = Image.new('L', (data.shape[1], data.shape[0] * 2), 255)
    # pixels = im.load()
    # for i in range(im.width):
    #     for j in range(data.shape[0]):
    #         pixels[i, j + data.shape[0]] = pixels[i, j] = int(data[j, i])
    # im = im.convert('RGB')
    # draw = ImageDraw.Draw(im, mode='RGB')
    # for i in range(len(text_column_record)):
    #     draw.line([(text_column_record[i], 0), (text_column_record[i], data.shape[0])], fill='#ff0000')
    # im.show()

    tmp_record = list()
    for i in range(len(text_column_record) - 1):
        if text_column_record[i + 1] - text_column_record[i] > 1:
            tmp_record.append((text_column_record[i] + 1, text_column_record[i + 1]))
    text_column_record = tmp_record
    # print(text_column_record)
    return text_column_record


def min_filter(data, size=3):  # block: size * size
    tmpdata = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # tmpdata[i, j] = np.min(data[(i - 1 >= 0) * (i - size // 2): i + size // 2, (j - 1 >= 0) * (j - size // 2): j + size // 2])
            tmpdata[i, j] = np.min(data[((i - size // 2) >= 0) * (i - size // 2): i + size // 2, ((j - size // 2) >= 0) * (j - size // 2): j + size // 2])
    return tmpdata


def max_filter(data, size=3):  # block: size * size
    tmpdata = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # tmpdata[i, j] = np.max(data[(i - 1 >= 0) * (i - size // 2): i + size // 2, (j - 1 >= 0) * (j - size // 2): j + size // 2])
            tmpdata[i, j] = np.max(data[((i - size // 2) >= 0) * (i - size // 2): i + size // 2 + 1, ((j - size // 2) >= 0) * (j - size // 2): j + size // 2 + 1])
    return tmpdata


def noise_fiilter(data, size=3):  # block: size * size
    tmpdata = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if tmpdata[i, j] == 0 and np.sum(data[((i - size // 2) >= 0) * (i - size // 2): i + size // 2 + 1, ((j - size // 2) >= 0) * (j - size // 2): j + size // 2 + 1]) >= 255 * 6:
                tmpdata[i, j] = 255
            else:
                tmpdata[i, j] = data[i, j]
    return tmpdata


def binarize(im):
    im = im.convert('L')
    pixels = im.load()
    data = np.zeros((im.height, im.width))
    for i in range(im.width):  # binarize
        for j in range(im.height):
            pixels[i, j] = 255 if pixels[i, j] > 165 else 0
            data[j, i] = pixels[i, j]
    return data


def extract_word_in_line(data, text_column_record, line_index):  # single line text
    dir = '/Users/Jeiel/Desktop/tmp/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(len(text_column_record)):
        im = create_im_with_data(data[:, text_column_record[i][0]: text_column_record[i][1]])
        im.save(dir + str(line_index) + '_' + str(i + 1) + '.bmp')
        # print(str(line_index) + '_' + str(i + 1) + ': ' + pytesseract.image_to_string(im, lang='chi_sim+eng'))


def create_im_with_data(data):  # gray
    im = Image.new('L', (data.shape[1], data.shape[0]))
    pixels = im.load()
    for i in range(im.width):
        for j in range(im.height):
            pixels[i, j] = int(data[j, i])
    return im

def main():
    im = Image.open('/Users/Jeiel/Dropbox/数字图像处理/实验/实验五-内容和素材/sample-24 copy.jpg')
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(2)
    data = binarize(im)

    # im = im.filter(ImageFilter.MinFilter(3))
    # im = im.filter(ImageFilter.MaxFilter(3))
    # im.show()
    # plt.figure()
    # plt.imshow(data, cmap="gray")
    # plt.show()
    # return
    text_line_record = split_text_line(im, data)
    # data = noise_fiilter(data)
    expand_data = data
    expand_data = min_filter(expand_data)
    # expand_data = min_filter(expand_data)
    # expand_data = max_filter(expand_data)
    # plt.figure()
    # plt.imshow(expand_data, cmap="gray")
    # plt.show()
    # return
    # plt.figure()
    # text_line = expand_data[text_line_record[0][0]: text_line_record[0][1]]
    # text_column_record = split_text_column(text_line)
    # text_line = data[text_line_record[0][0]: text_line_record[0][1]]
    # extract_word_in_line(text_line, text_column_record, 0)
    for i in range(len(text_line_record)):
        # plt.subplot(len(text_line_record), 1, i + 1)
        # plt.imshow(expand_data[text_line_record[i][0]: text_line_record[i][1]], cmap='gray')
        # print(text_line_record[i][1] - text_line_record[i][0])
        if text_line_record[i][1] - text_line_record[i][0] > 50:
            expand_data[text_line_record[i][0]: text_line_record[i][1]] = min_filter(expand_data[text_line_record[i][0]: text_line_record[i][1]])
        text_line = expand_data[text_line_record[i][0]: text_line_record[i][1]]
        text_column_record = split_text_column(text_line)
        text_line = data[text_line_record[i][0]: text_line_record[i][1]]
        extract_word_in_line(text_line, text_column_record, i + 1)
    # plt.show()


if __name__ == '__main__':
    main()
