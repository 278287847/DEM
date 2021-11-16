# -*- coding : utf-8 -*-
# coding: utf-8
"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

start_time = time.time()

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
import tensorflow as tf

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU selection"0,1" means choose 0# and 1# GPU
slim = tf.contrib.slim

tf.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string('checkpoint_path_inception_v3', '',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_v4', '',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_inception_resnet_v2',
                       '',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('checkpoint_path_resnet', '',
                       'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_string('dev_path', '', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

FLAGS = tf.flags.FLAGS


def gkern(kernlen=21, nsig=3):
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel15 = gkern(15, 3).astype(np.float32)

stack_kernel15 = np.stack([kernel15, kernel15, kernel15]).swapaxes(2, 0)
stack_kernel15 = np.expand_dims(stack_kernel15, 3)

def load_images(input_dir, batch_shape):

    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imresize(imread(f, mode='RGB'), [FLAGS.image_height, FLAGS.image_width]).astype(np.float) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):

    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 1001

    logit1, auxlogits1 = logit_inception_v3(input_diversity(x, 500))

    logit2, auxlogits2 = logit_inception_v4(input_diversity(x, 500))

    logit3, auxlogits3 = logit_inception_resnet(input_diversity(x, 500))

    logit4 = logit_resnet_v2(input_diversity(x, 500))

    logit = (logit1 + logit2 + logit3 + logit4) / 4
    auxlogits = (auxlogits1 + auxlogits2 + auxlogits3) / 3

    cross_entropy = tf.losses.softmax_cross_entropy(y,
                                                    logit,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(y,
                                                     auxlogits,
                                                     label_smoothing=0.0,
                                                     weights=0.4)

    noise = tf.gradients(cross_entropy, x)[0]
    noise = tf.nn.depthwise_conv2d(noise, stack_kernel15, strides=[1, 1, 1, 1], padding='SAME')
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + eps * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)

def logit_inception_v3(input_tensor):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(
            input_tensor, num_classes=1001, is_training=False)
    logits = logits
    auxlogits = end_points['AuxLogits']
    return logits, auxlogits

def logit_inception_v4(input_tensor):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, end_points = inception_v4.inception_v4(
            input_tensor, num_classes=1001, reuse=tf.AUTO_REUSE, is_training=False)
    logits = logits
    auxlogits = end_points['AuxLogits']
    return logits, auxlogits

def logit_inception_resnet(input_tensor):
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(
            input_tensor, num_classes=1001, reuse=tf.AUTO_REUSE, is_training=False)
    logits = logits
    auxlogits = end_points['AuxLogits']
    return logits, auxlogits

def logit_resnet_v2(input_tensor):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points = resnet_v2.resnet_v2_101(
            input_tensor, num_classes=1001, reuse=tf.AUTO_REUSE, is_training=False)
    logits = logits
    return logits


def input_diversity(input_tensor, resize_scale):
    rnd = tf.random_uniform((), FLAGS.image_width, resize_scale, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = resize_scale - rnd
    w_rem = resize_scale - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], resize_scale, resize_scale, 3))
    padded_resize = tf.image.resize_images(padded, [299, 299], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    padded_resize.set_shape((input_tensor.shape[0], 299, 299, 3))
    return padded_resize


def main(_):

    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = 1001
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

    tf.logging.set_verbosity(tf.logging.INFO)

    print(time.time() - start_time)

    with tf.Graph().as_default():

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            _, end_points = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)
        y = tf.one_hot(predicted_labels, num_classes)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
            s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
            s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
            s8.restore(sess, FLAGS.checkpoint_path_resnet)
            print(time.time() - start_time)

            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)

        print(time.time() - start_time)


if __name__ == '__main__':
    tf.app.run()
