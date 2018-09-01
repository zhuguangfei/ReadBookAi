# -*- coding:utf-8 -*-
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
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
            'weights', [192, 40], stddev=1 / 192.0, wd=None
        )
        biases = _variable_on_cpu('biases', [40], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
>>>>>>> 455d47956f6b55e2d067b84af43af96cb15c0c56
    return softmax_linear
