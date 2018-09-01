# -*- coding:utf-8 -*-
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer):
    var = tf.get_variable(
        name=name, shape=shape, dtype=tf.float16, initializer=initializer
    )
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0
        )
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    pool1 = tf.nn.max_pool(
<<<<<<< HEAD
        conv1, ksize=[1, 3, 3, 1], padding='SAME', name='pool1')

    norml1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 /
                       9.0, beta=0.75, name='norml1')
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norml1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    norml2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 /
                       9.0, beta=0.75, name='norml2')
    pool2 = tf.nn.max_pool(norml2, ksize=[1, 3, 3, 1], strides=[
                           1, 2, 2, 1], padding='SAME', name='pool2')
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [32, dim])

        weights = _variable_with_weight_decay(
            'weights', shape=[dim, 384], stddev=0.04, wd=0.004)

        biases = _variable_on_cpu(
            'biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                            biases, name=scope.name)

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[382, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu(
            'biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights)+biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', [192, 100], stddev=1/192, wd=0.0)
        biases = _variable_on_cpu(
            'biases', [100], tf.constant_initializer(0.0))
        softmax_linear = tf.add(
            tf.matmul(local4, weights), biases, name=scope.name)
=======
        conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1'
    )
    norml1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norml1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=None
        )
        conv = tf.nn.conv2d(norml1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    norml2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norml2')
    pool2 = tf.nn.max_pool(
        norml2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2'
    )

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay(
            'weights', shape=[dim, 384], stddev=0.04, wd=0.004
        )
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights', shape=[384, 192], stddev=0.04, wd=0.004
        )
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', [192, 80], stddev=1 / 192.0, wd=None
        )
        biases = _variable_on_cpu('biases', [80], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
>>>>>>> 455d47956f6b55e2d067b84af43af96cb15c0c56
    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


from data_util import get_xy, get_xy_test
import numpy as np


def train():
    training_epochs = 10
    batch_size = 10
    X, Y = get_xy()
    train_x = tf.placeholder(tf.float16, [batch_size, 500, 500, 3])
    train_y = tf.placeholder(tf.float16, [batch_size, 80])
    softmax_linear = inference(train_x)
    total_loss = loss(softmax_linear, train_y)
    tf.train.AdamOptimizer(0.001).minimize(total_loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(training_epochs):
            # for _ in range(10):
            print('--------------')
            indexs = np.random.choice(range(10), 10)
            batch_x = X[indexs]
            batch_y = Y[indexs]
            ls = sess.run([total_loss], feed_dict={train_x: batch_x, train_y: batch_y})
            saver.save(sess, 'save/checkpoint', global_step=step)
            print(ls)


def predict():
    X, Y = get_xy_test()
    predict_x = tf.placeholder(tf.float16, [1, 500, 500, 3])
    softmax_linear = inference(predict_x)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('save')
        saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.global_variables_initializer())
        softmax_linear = sess.run([softmax_linear], {predict_x: X})
        print(softmax_linear)
        # for s in softmax_linear:
        #     a = 0
        #     for i in s:
        #         a += i
        #     print(a)


if __name__ == '__main__':
    # train()
    predict()
