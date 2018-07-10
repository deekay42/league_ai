from tflearn.layers.conv import highway_conv_2d
from tflearn.layers.estimator import regression

import tflearn
# Data loading and preprocessing
from tflearn.activations import relu

from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization



SPELL_IMG_SIZE = (20,20)
ITEM_IMG_SIZE = (24,24)
CHAMP_IMG_SIZE = (20,20)
SELF_IMG_SIZE = (20,20)
CORNER_IMG_SIZE = (400,300)
COORDS_IMG_SIZE = (300, 122)


def detect_corners(img_size):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, *img_size, 3], name='input')

    conv1a_3_3 = relu(batch_normalization(
        conv_2d(network, 32, 3, strides=2, bias=False, padding='VALID', activation=None, name='Conv2d_1a_3x3'), trainable=is_training))
    conv2a_3_3 = relu(batch_normalization(
        conv_2d(conv1a_3_3, 32, 3, bias=False, padding='VALID', activation=None, name='Conv2d_2a_3x3'), trainable=is_training))
    conv2b_3_3 = relu(
        batch_normalization(conv_2d(conv2a_3_3, 64, 3, bias=False, activation=None, name='Conv2d_2b_3x3'), trainable=is_training))
    maxpool3a_3_3 = max_pool_2d(conv2b_3_3, 3, strides=2, padding='VALID', name='MaxPool_3a_3x3')
    conv3b_1_1 = relu(batch_normalization(
        conv_2d(maxpool3a_3_3, 80, 1, bias=False, padding='VALID', activation=None, name='Conv2d_3b_1x1'), trainable=is_training))
    conv4a_3_3 = relu(batch_normalization(
        conv_2d(conv3b_1_1, 192, 3, bias=False, padding='VALID', activation=None, name='Conv2d_4a_3x3'), trainable=is_training))
    maxpool5a_3_3 = max_pool_2d(conv4a_3_3, 3, strides=2, padding='VALID', name='MaxPool_5a_3x3')

    tower_conv = relu(
        batch_normalization(conv_2d(maxpool5a_3_3, 96, 1, bias=False, activation=None, name='Conv2d_5b_b0_1x1'), trainable=is_training))

    tower_conv1_0 = relu(
        batch_normalization(conv_2d(maxpool5a_3_3, 48, 1, bias=False, activation=None, name='Conv2d_5b_b1_0a_1x1'), trainable=is_training))
    tower_conv1_1 = relu(
        batch_normalization(conv_2d(tower_conv1_0, 64, 5, bias=False, activation=None, name='Conv2d_5b_b1_0b_5x5'), trainable=is_training))

    tower_conv2_0 = relu(
        batch_normalization(conv_2d(maxpool5a_3_3, 64, 1, bias=False, activation=None, name='Conv2d_5b_b2_0a_1x1'), trainable=is_training))
    tower_conv2_1 = relu(
        batch_normalization(conv_2d(tower_conv2_0, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0b_3x3'), trainable=is_training))
    tower_conv2_2 = relu(
        batch_normalization(conv_2d(tower_conv2_1, 96, 3, bias=False, activation=None, name='Conv2d_5b_b2_0c_3x3'), trainable=is_training))

    tower_pool3_0 = avg_pool_2d(maxpool5a_3_3, 3, strides=1, padding='same', name='AvgPool_5b_b3_0a_3x3')
    tower_conv3_1 = relu(
        batch_normalization(conv_2d(tower_pool3_0, 64, 1, bias=False, activation=None, name='Conv2d_5b_b3_0b_1x1'), trainable=is_training))

    tower_5b_out = merge([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1], mode='concat', axis=3)

    net = fully_connected(tower_5b_out, 128, activation='relu')
    # net = dropout(net, 0.5)
    # output
    net = fully_connected(net, 8, activation='linear')
    # net = dropout(net, 0.5)
    # output

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='mean_square', name='target', metric='R2')

def detect_cornersV2(img_size):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[1], img_size[0], 1], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 5, bias=False, activation=None), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 2)

    conv2 = relu(batch_normalization(
        conv_2d(maxpool1, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool2 = max_pool_2d(conv2, 2)

    conv3 = relu(batch_normalization(
        conv_2d(maxpool2, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool3 = max_pool_2d(conv3, 2)

    conv4 = relu(batch_normalization(
        conv_2d(maxpool3, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool4 = max_pool_2d(conv4, 2)

    conv5 = relu(batch_normalization(
        conv_2d(maxpool4, 32, 5, bias=False, activation=None), trainable=is_training))
    maxpool5 = max_pool_2d(conv5, 2)

    net = fully_connected(maxpool5, 512, activation='relu')
    net = fully_connected(net, 8, activation='linear')

    return regression(net, optimizer='nesterov', learning_rate=learning_rate,
                      loss='mean_square', name='target', metric='R2')

def classify_spells(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[1], img_size[0], 3], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 3, bias=False, activation=None), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 3)

    conv2 = relu(batch_normalization(
        conv_2d(maxpool1, 32, 3, bias=False, activation=None), trainable=is_training))
    maxpool2 = max_pool_2d(conv2, 3)

    conv3 = relu(batch_normalization(
        conv_2d(maxpool2, 32, 3, bias=False, activation=None), trainable=is_training))
    maxpool3 = max_pool_2d(conv3, 3)

    net = fully_connected(maxpool3, 128, activation='relu')
    net = fully_connected(net, num_elements, activation='softmax')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')

def classify_champs(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[1], img_size[0], 1], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 3)

    conv2 = relu(batch_normalization(
        conv_2d(maxpool1, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    maxpool2 = max_pool_2d(conv2, 3)

    net = fully_connected(maxpool2, 64, activation='relu', regularizer="L2")
    # net = fully_connected(net, 128, activation='relu')
    net = fully_connected(net, num_elements, activation='softmax')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')


def classify_items(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[0], img_size[1], 3], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    # maxpool1 = max_pool_2d(conv1, 2)

    conv2 = relu(batch_normalization(
        conv_2d(conv1, 32, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    # maxpool2 = max_pool_2d(conv2, 2)

    # conv3 = relu(batch_normalization(
    #     conv_2d(conv2, 64, 3, bias=False, activation=None, regularizer="L2"), trainable=is_training))

    net = fully_connected(conv2, 128, activation='relu', regularizer="L2")
    net = fully_connected(net, num_elements, activation='softmax')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='categorical_crossentropy', name='target')


def classify_self(img_size, num_elements, learning_rate):

    is_training = tflearn.get_training_mode()
    # input
    network = input_data(shape=[None, img_size[0], img_size[1], 1], name='input')

    conv1 = relu(batch_normalization(
        conv_2d(network, 8, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    maxpool1 = max_pool_2d(conv1, 2)

    # conv2 = relu(batch_normalization(
    #     conv_2d(maxpool1, 16, 5, bias=False, activation=None, regularizer="L2"), trainable=is_training))
    # maxpool2 = max_pool_2d(conv2, 2)

    # net = fully_connected(maxpool2, 16, activation='relu', regularizer="L2")
    net = fully_connected(maxpool1, num_elements, activation='sigmoid')

    return regression(net, optimizer='adam', learning_rate=learning_rate,
                      loss='binary_crossentropy', name='target')
