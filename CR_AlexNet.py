# /usr/bin/python3
# __*__ coding: utf-8 __*__

from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
# from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.data_utils import build_hdf5_image_dataset
from tflearn.data_utils import image_preloader

# dataset_file = 'num.txt'
dataset_file = 'samples_109.txt'
catergory = 109
# build_hdf5_image_dataset(dataset_file, image_shape=(18, 18), mode='file', output_path='dataset.h5', categorical_labels=True)
X, Y = image_preloader(dataset_file, image_shape=(18, 18), mode='file', categorical_labels=True, normalize=True)
# print(np.array(X).shape)
print(np.reshape(X, (-1, 18, 18, 1)).shape)
X = np.reshape(X, (-1, 18, 18, 1))

# Building 'AlexNet'
network = input_data(shape=[None, 18, 18, 1])
network = conv_2d(network, 96, 3, strides=1, activation='relu')
network = max_pool_2d(network, 3, strides=1)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=1)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=1)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.7)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.7)
network = fully_connected(network, catergory, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.01)
# print(network)
# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=30, validation_set=(X, Y), shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          # snapshot_epoch=False,
          run_id='alexnet_momentum')
model.save('cr_alexnet.tflearn')
