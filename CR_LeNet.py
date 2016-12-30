# /usr/bin/python3
# __*__ coding: utf-8 __*__


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader
import numpy as np


dataset_file = 'samples_109.txt'
catergory = 109
X, Y = image_preloader(dataset_file, image_shape=(18, 18), mode='file', categorical_labels=True, normalize=True, )
# print(np.array(X).shape)
print(np.reshape(X, (-1, 18, 18, 1)).shape)
X = np.reshape(X, (-1, 18, 18, 1))

# Building convolutional network
network = input_data(shape=[None, 18, 18, 1], name='input')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 512, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, catergory, activation='softmax')
network = regression(network, optimizer='momentum', learning_rate=0.05,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=1000,
           validation_set=({'input': X}, {'target': Y}),
           snapshot_step=100, batch_size=X.shape[0], show_metric=True, run_id='CR_LeNet_1000_double_size')
model.save('LeNet_model_1000_double_size.tflearn')
