# /usr/bin/python3
# __*__ coding: utf-8 __*__


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import operator


def getImageData(filename, size=(18, 18)):  # size:(width, height)
    im = Image.open(filename).resize(size)
    data = np.array(im).reshape((1, *size, 1))
    return data


def load_Image(dir):
    im_dict = dict()
    for x in os.listdir(path=dir):
        x = os.path.join(dir, x)
        if os.path.isfile(x) and os.path.splitext(x)[1] == '.bmp':
            im_dict[x] = getImageData(x)
    im_dict = sorted(im_dict.items(), key=operator.itemgetter(0))
    return im_dict


def init_vocabulary():
    vocabulary_file = '/Users/Jeiel/github/dipexpt/vocabulary.txt'
    vocabulary = dict()
    with open(vocabulary_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            vocabulary[line.rsplit()[1]] = line.rsplit()[0]
    return vocabulary


catergory = 109

# Building convolutional network
network = input_data(shape=[None, 18, 18, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, catergory, activation='softmax')
network = regression(network, optimizer='momentum', learning_rate=0.05,
                     loss='categorical_crossentropy', name='target')

# prediction
model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('LeNet_Model/LeNet_model_1000.tflearn')
# data = getImageData('/Users/Jeiel/Desktop/tmp/1_1.bmp')
# print(np.argmax(model.predict(data.reshape((1, 18, 18, 1)))))

vocabulary = init_vocabulary()
split_text_image_dir = '/Users/Jeiel/Desktop/tmp/'
images_dict = load_Image(split_text_image_dir)
for item in images_dict:
    print(item[0], vocabulary[str(np.argmax(model.predict(item[1])))])
