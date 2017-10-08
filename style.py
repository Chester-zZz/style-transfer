import tensorflow as tf
import scipy
import os
import argparse
import utils
import vgg
import transform
import time
from functools import reduce


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_images_dir', type=str, help='image you want to transfer')
    parser.add_argument('--style_image_path', type=str, help='style image full path')
    parser.add_argument('--output_dir', type=str, help='output_dir')
    parser.add_argument('--vgg_path', type=str, help='vgg19 model path')
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="directory with checkpoint to resume")
    parser.add_argument('--epoch_num', type=int, help='epochs of computation')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--content_weight', type=float, help='content weight of the result', default=0.15)
    parser.add_argument('--style_weight', type=float, help='style weight of the result', default=100.0)
    parser.add_argument('--tv_weight', type=float, help='total variance weight of the result', default=200.0)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--summary_freq', type=int, help='write summary every this steps', default=10)
    parser.add_argument('--save_image_freq', type=int, help='write summary every this steps', default=1000)
    parser.add_argument('--save_model_freq', type=int, help='save model every this steps', default=10)
    parser.add_argument('--model_max_to_keep', type=int, help='max num of models to keep', default=5)
    return parser


def main():
    vgg_layers = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    ]
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    CONTENT_LAYER = ['relu4_2']
    # 解析参数
    options = build_parser().parse_args()
    print(options)
    if not os.path.exists(options.output_dir):
        os.mkdir(options.output_dir)

    style_Graph = tf.Graph()
    content_Graph = tf.Graph()

    # 由于style的features只需要计算一次，所以先把结果计算出来避免重复计算
    with style_Graph.as_default():
        with tf.name_scope('style'):
            # 加载风格图片数据，得到batch
            _, style_path, style_input = utils.get_input_data(options.style_image_path, [256, 256, 3], 1, kind='style')
            # print(style_input)
            with tf.variable_scope('vgg19'):
                # 将风格图片数据放到vgg抽features
                style_features = vgg.build_vgg(options.vgg_path, vgg_layers, style_input)
            # 计算style_image得到的风格Gram矩阵
            style_grams = {}
            for style_layer_1 in STYLE_LAYERS:
                # 注意这里是batch
                style_grams[style_layer_1] = utils.calculate_Grams(style_features[style_layer_1])
        sv_1 = tf.train.Supervisor(logdir=options.output_dir, save_summaries_secs=0, saver=None)
        with sv_1.managed_session() as sess_1:
            # init_style = tf.global_variables_initializer()
            # sess_1.run(init_style)

            # 如果直接写sess_1.run不行，好像是读取queue的线程没启动，得加上这么一句
            # 但是加上后面没关闭线程又会报错，索性直接用Supervisor的写法好了
            # tf.train.start_queue_runners(sess_1)

            # print(sess_1.run(style_features['relu1_1']))
            # for style_layer_1 in STYLE_LAYERS:
            #     print(style_layer_1)
            # print(style_grams[style_layer_1])
            style_grams_result = sess_1.run(style_grams)
            print('style grams calculation finish')

    with content_Graph.as_default():

        # 将训练数据放到vgg中
        with tf.name_scope('content'):
            # 加载训练数据，得到batch
            images_count, content_paths, content_input = utils.get_input_data(options.content_images_dir, [256, 256, 3], options.batch_size, kind='content')
            with tf.variable_scope('vgg19'):
                # 将训练数据放到vgg抽features
                content_features = vgg.build_vgg(options.vgg_path, vgg_layers, content_input)

        # 将训练数据放到transform网络中，并将结果放到vgg中
        with tf.name_scope('transform'):
            # 将训练数据放到transform网络中得到输出
            content_t = transform.build_transform(content_input)
            content_t_for_output = tf.image.convert_image_dtype((content_t + 1) / 2, dtype=tf.uint8, saturate=True)
            tf.summary.image('transform_result', content_t_for_output)
            with tf.variable_scope('vgg19', reuse=True):
                # 再将transform的输出放到vgg里面抽features
                content_t_features = vgg.build_vgg(options.vgg_path, vgg_layers, content_t)

            # 计算训练数据得到的风格Gram矩阵
            content_t_grams = {}
            for style_layer_1 in STYLE_LAYERS:
                # print(style_layer_1)
                content_t_grams[style_layer_1] = utils.calculate_Grams(content_t_features[style_layer_1])

        # 定义style损失
        with tf.name_scope('style_loss'):
            style_losses = []
            for style_layer_1 in style_grams_result:
                style_gram = style_grams_result[style_layer_1]
                # 后面除那个size是为了每一层得到的loss都差不多（因为channel数不一样），归一化
                style_losses.append(tf.nn.l2_loss(content_t_grams[style_layer_1] - style_gram) / style_gram.size)
            # 注意这里是为了让每一次训练，虽然batch_size不同，但loss都差不多，方便观察
            style_loss = options.style_weight * 2 * reduce(tf.add, style_losses) / options.batch_size
            tf.summary.scalar('style_loss', style_loss)

        # 定义content损失
        with tf.name_scope('content_loss'):
            content_losses = []
            for content_layer_1 in CONTENT_LAYER:
                content_size = utils.get_size(content_t_features[content_layer_1])
                content_losses.append(tf.nn.l2_loss(content_t_features[content_layer_1]
                                                    - content_features[content_layer_1]) / content_size)

            content_loss = options.content_weight * 2 * reduce(tf.add, content_losses) / options.batch_size
            tf.summary.scalar('content_loss', content_loss)
        # print(1111111111111)
        # 定义total variance损失，和图像的平滑度有关，其实就是梯度图
        with tf.name_scope('tv_loss'):
            content_shape = content_t.get_shape()
            content_t_x_shape = int(content_shape[2])
            content_t_y_shape = int(content_shape[1])
            content_t_x_size = utils.get_size(content_t[:, :, 1:, :])
            content_t_y_size = utils.get_size(content_t[:, 1:, :, :])
            tv_x = tf.nn.l2_loss(content_t[:, :, 1:, :] - content_t[:, :, :content_t_x_shape - 1, :]) / content_t_x_size
            tv_y = tf.nn.l2_loss(content_t[:, 1:, :, :] - content_t[:, :content_t_y_shape - 1, :, :]) / content_t_y_size
            tv_loss = options.tv_weight * 2 * (tv_x + tv_y) / options.batch_size
            tf.summary.scalar('tv_loss', tv_loss)

        # 定义总损失
        with tf.name_scope('total_loss'):
            total_loss = style_loss + content_loss + tv_loss
            tf.summary.scalar('total_loss', total_loss)

        # 定义训练
        with tf.name_scope('train'):
            total_train = tf.train.AdamOptimizer(options.lr).minimize(total_loss)

        # 合并、定义summary
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(options.output_dir + '/train')

        # 定义图片保存
        content_t_for_save = tf.image.encode_jpeg(content_t_for_output)[0]

        # 总步数
        max_steps = int(options.epoch_num * images_count / options.batch_size)

        # 模型保存
        saver = tf.train.Saver(max_to_keep=options.model_max_to_keep)
        # 初始化
        sv_2 = tf.train.Supervisor(logdir=options.output_dir, save_summaries_secs=0, saver=None)
        with sv_2.managed_session() as sess:
            if options.checkpoint is not None:
                print('Load model from latest checkpoint...')
                checkpoint = tf.train.latest_checkpoint(options.checkpoint)
                saver.restore(sess, checkpoint)
            start_time = time.time()
            # 循环train
            for step in range(max_steps):
                print('step: ', step)
                sess.run(total_train)

                # 保存summary
                if utils.should(options.summary_freq, step, max_steps):
                    print('Summary...')
                    average_step_time = (time.time() - start_time) / step
                    time_need = int((max_steps - step) * average_step_time / 60.0) + 1
                    print('still need %d minutes to finish...' % time_need)
                    summary_result = sess.run(merged_summary)
                    train_writer.add_summary(summary_result, step)

                if utils.should(options.save_model_freq, step, max_steps):
                    print('Save model...')
                    saver.save(sess, os.path.join(options.output_dir, 'model'), global_step=step)


main()
# python style.py --content_images_dir F:\projects\python\tf_test\style-transfer\images\train_images --style_image_path F:\projects\python\tf_test\style-transfer\images\style.jpg --output_dir F:\projects\python\tf_test\style-transfer\output --vgg_path F:\dl\imagenet-vgg-verydeep-19.mat --epoch_num 100 --batch_size 1
