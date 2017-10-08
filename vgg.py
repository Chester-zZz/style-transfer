import tensorflow as tf
import numpy as np
import scipy.io
# layers = [
#         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
#         'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
#         'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
#         'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
#          ]


def build_vgg(vgg_mat_path, layers_list,input_data):
    vgg_data = scipy.io.loadmat(vgg_mat_path)
    layers_data = vgg_data['layers'][0]
    net = {}
    # 这样用placeholder，就建立一个vgg就行了，run的时候改变输入就行
    current = input_data
    for i, name in enumerate(layers_list):
        kind = name[:4]
        with tf.variable_scope(name):
            if kind == 'conv':
                weights, bias = layers_data[i][0][0][0][0]
                weights = np.transpose(weights, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                
                # 让vgg的参数不更新
                weights_t = tf.Variable(initial_value=weights, trainable=False, name='weights_t')
                bias_t = tf.Variable(initial_value=bias, trainable=False, name='bias_t')
                current = tf.nn.conv2d(current, weights_t, strides=[1, 1, 1, 1], padding='SAME')
                current = tf.nn.bias_add(current, bias_t)
            if kind == 'relu':
                current = tf.nn.relu(current)
            if kind == 'pool':
                current = tf.nn.max_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net[name] = current
    assert len(net) == len(layers_list)
    return net


def preprocess(input_image):
    mean = np.array([123.68, 116.779, 103.939])
    return input_image - mean


def depreprocess(output_image):
    mean = np.array([123.68, 116.779, 103.939])
    return input_image + mean
