# dcgan by liming @17.7.10
"""
Train 1.0

Usage:
    train.py device <device> d1 <d1> d2 <d2>

Options:
	-h --help
"""


import os
import sys
from docopt import docopt
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)

import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob

from model import *
from pre_data import *
import datetime

Image_h = 96
Image_w = 96
Sample_num = 30000
Epoch_num = 400

Batch_size = 128
G_learnrate = 1e-3
D_learnrate = 1e-3

# Data_dir = 'faces'
Data_dir1 = '/celeba/'
Data_dir2 = '/faces/'
Data_pattern1 = '*.jpg'
Data_pattern2 = '*.jpg'


def optimizer(loss, learning_rate, vlist=None, name=None):
    with tf.variable_scope(name):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name=name + '/Adam')
        return opt.minimize(loss, var_list=vlist, name=name + '/opt')


def draw_img(x):
    pass


def __main__(d1, d2, cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    D1_LOSS = float(d1)
    D2_LOSS = float(d2)
    ti = datetime.datetime.now()
    log_dir = (
        'log/' + str(D1_LOSS) + '_' + str(D2_LOSS) + '/' + str(ti.month) + '-' + str(ti.day) + '-' + str(
            ti.hour) + '-' + str(ti.minute)
        + '-' + str(ti.second) + '/')
    tensorboad_dir = log_dir

    # noise input
    noise_input = tf.placeholder(tf.float32, shape=[None, 100], name='noise')
    noise_sample_input = tf.placeholder(tf.float32, shape=[None, 100], name='noise')
    # real data input
    image_input1 = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, 3], name='image1')
    image_input2 = tf.placeholder(tf.float32, shape=[None, Image_h, Image_w, 3], name='image2')

    # generate G
    G = generate(noise_input, Image_h, Image_w, True, None, batch_size=Batch_size)
    # param of G
    G_vars = tf.trainable_variables()

    G_sample = generate(noise_sample_input, Image_h, Image_w, False, True, batch_size=Batch_size)
    img_sample = restruct_image(G_sample, Batch_size)
    tf.summary.image('generated image', img_sample, Batch_size)
    # decrim
    D1 = decrim1(image_input1, True, None, batch_size=Batch_size)
    # param of d


    D2 = decrim2(image_input2, True, None, batch_size=Batch_size)
    # D_vars = []
    # for item in tf.trainable_variables():
    # 	if item not in G_vars:
    # 		D_vars.append(item)
    # param of d
    # D2_vars = []
    # for item in tf.trainable_variables():
    # 	if item not in G_vars:
    # 		if item not in D1_vars:
    # 			D2_vars.append(item)

    d1_real = D1
    d2_real = D2

    d1_fake = decrim1(G, True, True)
    d2_fake = decrim2(G, True, True)

    loss_train_D1 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_real, labels=tf.ones_like(d1_real))) \
                    + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_fake, labels=tf.zeros_like(d1_fake)))
    tf.summary.scalar('d1_loss', loss_train_D1)

    loss_train_D2 = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_real, labels=tf.ones_like(d2_real))) \
                    + tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_fake, labels=tf.zeros_like(d2_fake)))
    tf.summary.scalar('d2_loss', loss_train_D2)

    # loss_train_D = loss_train_D1 + loss_train_D2
    # tf.summary.scalar('d_loss',loss_train_D)

    loss_train_G = D1_LOSS * tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_fake, labels=tf.ones_like(d1_fake))) \
                   + D2_LOSS * tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_fake, labels=tf.ones_like(d2_fake)))
    tf.summary.scalar('g_loss', loss_train_G)
    # loss_train_G = d_fake
    # loss_train_D = -(d_real + d_fake)
    # loss_train_G = (1 / 2) * (d_fake - 1) ** 2
    # loss_train_D = (1 / 2) * (d_real - 1) ** 2 + (1 / 2) * (d_fake) ** 2
    g_optimizer = optimizer(loss_train_G, G_learnrate, G_vars, name='opt_train_G')
    d1_optimizer = optimizer(loss_train_D1, D_learnrate, name='opt_train_D1')
    d2_optimizer = optimizer(loss_train_D2, D_learnrate, name='opt_train_D2')

    # noise_sample = np.random.uniform(-1,1,[Batch_size,100]).astype('float32')
    # ==============================Start training=============================
    with tf.Session() as sess:
        # =====tensorboard=============
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(tensorboad_dir, sess.graph)
        # =============================
        sess.run(tf.global_variables_initializer())
        image_list1 = get_datalist(Data_dir1, Data_pattern1)[:Sample_num]
        image_list2 = get_datalist(Data_dir2, Data_pattern2)[:Sample_num]
        image_len = int(len(image_list1))
        batch_num = int(image_len / Batch_size)
        count = 0
        for e in range(Epoch_num):

            for idx in range(batch_num):
                # prepare data
                z = np.random.normal(0, 1, [Batch_size, 100]).astype('float32')
                img_batch_list1 = image_list1[idx * Batch_size:(idx + 1) * Batch_size]
                img_batch1 = get_image(img_batch_list1, Batch_size, Image_h, Image_w)

                img_batch_list2 = image_list2[idx * Batch_size:(idx + 1) * Batch_size]
                img_batch2 = get_image(img_batch_list2, Batch_size, Image_h, Image_w)

                _, d1_loss = sess.run([d1_optimizer, loss_train_D1],
                                      feed_dict={
                                          noise_input: z,
                                          image_input1: img_batch1,

                                      })
                _, d2_loss = sess.run([d2_optimizer, loss_train_D2],
                                      feed_dict={
                                          noise_input: z,
                                          image_input2: img_batch2,

                                      })

                _, g_loss = sess.run([g_optimizer, loss_train_G],
                                     feed_dict={
                                         noise_input: z,
                                         image_input1: img_batch1,
                                         image_input2: img_batch2,

                                     })

                _, g_loss = sess.run([g_optimizer, loss_train_G],
                                     feed_dict={
                                         noise_input: z,
                                         image_input1: img_batch1,
                                         image_input2: img_batch2,

                                     })

                print("epoch: %d batch: %d  gloss:%.4f d1loss:%.4f d2loss:%.4f" %
                      (e + 1, idx, g_loss, d1_loss, d2_loss))

                if idx % 20 == 0:
                    count = count + 1
                    noise_sample = np.random.normal(0, 1, [Batch_size, 100]).astype('float32')

                    summary_all = sess.run(merged_summary_op, feed_dict={
                        noise_sample_input: noise_sample,
                        noise_input: z,
                        image_input1: img_batch1,
                        image_input2: img_batch2,

                    })
                    summary_writer.add_summary(summary_all, count)


# =================================================================
if __name__ == "__main__":
    arguments = docopt(__doc__)
    d1 = arguments["<d1>"]
    d2 = arguments["<d2>"]
    cuda = arguments["<device>"]
    __main__(d1, d2, cuda)
