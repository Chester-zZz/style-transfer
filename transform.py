'''
build风格转换网络，也就是主要训练的网络
'''

import tensorflow as tf


def build_transform(input_tensor):
    with tf.variable_scope('trans_net'):
        with tf.variable_scope('conv_1'):
            conv1 = conv(input_tensor, 9, 32, 1)
        with tf.variable_scope('conv_2'):
            conv2 = conv(conv1, 3, 64, 2)
        with tf.variable_scope('conv_3'):
            conv3 = conv(conv2, 3, 128, 2)
        with tf.variable_scope('residual_1'):
            resi1 = residual_block(conv3, 3, 128, 1)
        with tf.variable_scope('residual_2'):
            resi2 = residual_block(resi1, 3, 128, 1)
        with tf.variable_scope('residual_3'):
            resi3 = residual_block(resi2, 3, 128, 1)
        with tf.variable_scope('residual_4'):
            resi4 = residual_block(resi3, 3, 128, 1)
        with tf.variable_scope('residual_5'):
            resi5 = residual_block(resi4, 3, 128, 1)
        with tf.variable_scope('deconv_1'):
            deconv1 = deconv(resi5, 3, 64, 2)
        with tf.variable_scope('deconv_2'):
            deconv2 = deconv(deconv1, 3, 32, 2)
        with tf.variable_scope('last_conv'):
            result = conv(deconv2, 9, 3, 1, if_relu=False)
            result = tf.nn.tanh(result)
        # 这样得到的结果是在(-1,1)之间
        return result



def conv(input_tensor, filter_size, output_channels, stride, if_relu=True, if_bn=True):
    with tf.variable_scope('conv'):
        input_channels = input_tensor.get_shape()[3]
        filter = tf.get_variable('filter',
                                 [filter_size, filter_size, input_channels, output_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        result = tf.nn.conv2d(
            input_tensor, filter, [1, stride, stride, 1], padding='SAME')
    if if_relu:
        with tf.name_scope('relu'):
            result = tf.nn.relu(result)
    if if_bn:
        batch_norm(result)
    return result


def deconv(input_tensor, filter_size, output_channels, stride, if_relu=True, if_bn=True):
    with tf.variable_scope('deconv'):
        batch_num, input_height, input_width, input_channels = [int(d) for d in input_tensor.get_shape()]
        filter = tf.get_variable('filter', [filter_size, filter_size, output_channels, input_channels],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # print('decove---',[batch_num, input_height * 2, input_width * 2, output_channels])
        output_shape = [batch_num, input_height * stride, input_width * stride, output_channels]
        result = tf.nn.conv2d_transpose(input_tensor, filter, output_shape,
                                        [1, stride, stride, 1], padding="SAME")
    if if_relu:
        with tf.name_scope('relu'):
            result = tf.nn.relu(result)
    if if_bn:
        batch_norm(result)
    return result


def batch_norm(input_tensor):
    # 构造batch_normalization层
    with tf.variable_scope('batch_norm'):
        input_channels = input_tensor.get_shape()[3]
        gamma = tf.get_variable('bn_gamma',
                                [input_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1, 0.02))
        beta = tf.get_variable(
            'bn_beta', [input_channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        epsilon = 1e-5
        mean, variance = tf.nn.moments(input_tensor, [0, 1, 2], keep_dims=False)
        result = tf.nn.batch_normalization(
            input_tensor, mean, variance, beta, gamma, epsilon)
        return result


def residual_block(input_tensor, filter_size, output_channels, stride):
    with tf.variable_scope('resi_conv1'):
        conv1 = conv(input_tensor, filter_size, output_channels, stride)
    with tf.variable_scope('resi_conv2'):
        conv2 = conv(conv1, filter_size, output_channels, stride, if_relu=False)
    return tf.nn.relu(conv2 + input_tensor)
