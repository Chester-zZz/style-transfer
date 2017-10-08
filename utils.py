import os
import glob
import tensorflow as tf


def get_input_data(path, data_shape, batch_size, kind):
    if not path or not os.path.exists(path):
        raise Exception('input dir does not exist')
    if kind == 'style':
        input_paths = [path]
    if kind == 'content':
        input_paths = glob.glob(os.path.join(path, '*.jpg'))
    with tf.name_scope('load_images'):
        path_queue = tf.train.string_input_producer(input_paths)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = tf.image.decode_jpeg(contents)
        # 将图片数据归一化到(0,1)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        raw_input.set_shape(data_shape)
        # 归一化到(-1,1), 注意这还是一个tensor
        data_for_input = raw_input * 2.0 - 1
        path_batch, input_batch = tf.train.batch([paths, data_for_input], batch_size=batch_size)
        return len(input_paths), path_batch, input_batch


def calculate_Grams(input_batch):
    # 注意这里的input是batch，map_fn要求输入和输出有一定的格式，很麻烦，不用了
    batch_size, height, width, channels = [int(d) for d in input_batch.get_shape()]
    size = height * width * channels
    a = tf.reshape(input_batch, (batch_size, height * width, channels))
    a_T = tf.transpose(a, perm=[0, 2, 1])
    return tf.matmul(a_T, a) / size

def get_size(input_tensor):
    # 获取一个tensor的size，就是height * width * channels，注意，不包括batch_size
    f_bn, f_h, f_w, f_ch = [int(d) for d in input_tensor.get_shape()]
    return f_h * f_w * f_ch


def should(freq, step, max_steps):
    return freq > 0 and ((step + 1) % freq == 0 or step + 1 == max_steps)
