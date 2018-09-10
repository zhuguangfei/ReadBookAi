# -*- coding:utf-8 -*-
import tensorflow as tf

from data_util import get_xy


def create_variable(name, shape, initializer):
    var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def create_variable_with_weight_decay(name, shape, stddev, wd):
    var = create_variable(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def build_graph(batch_size, width, height, channel):
    input_x = tf.placeholder(tf.int8, [batch_size, height, width, channel])
    with tf.variable_scope('conv1') as scope:
        kernel = create_variable_with_weight_decay(
            'weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0
        )
        conv = tf.nn.conv2d(input_x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = create_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], padding='SAME')

    norml1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norml1')

    with tf.variable_scope('conv2') as scope:
        kernel = create_variable_with_weight_decay(
            'weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0
        )
        conv = tf.nn.conv2d(norml1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = create_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    norml2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norml2')
    pool2 = tf.nn.max_pool(norml2, ksize=[1, 3, 3, 1], padding='SAME')

    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool1.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [32, dim])
        weights = create_variable_with_weight_decay(
            'weights', shape=[dim, 384], stddev=1e-4, wd=0.004
        )
        biases = create_variable('biases', [384], tf.constant_initializer(0.0))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('local4') as scope:
        weights = create_variable_with_weight_decay(
            'weights', shape=[384, 192], stddev=1e-4, wd=0.004
        )
        biases = create_variable('biases', [192], tf.constant_initializer(0.0))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    with tf.variable_scope('softmax_linear') as scope:
        weights = create_variable_with_weight_decay(
            'weights', [192, 35], stddev=1 / 192, wd=0.0
        )
        biases = create_variable('biases', [35], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights) + biases)
    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


batch_size = 32
height = 250
weigth = 250
channel = 3
max_len = 35
words_len = 2022
learning_rate = 0.001
train_epochs = 10


def train():
    X, Y = get_xy()
    train_x = tf.placeholder(tf.float64, [batch_size, height, weigth, channel])
    train_y = tf.placeholder(tf.float64, [batch_size, max_len, words_len])
    softmax_linear = build_graph(batch_size, height, weigth, channel)
    loss_mean = loss(softmax_linear, train_y)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(
        loss_mean
    )
    savedir = 'save'
    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_epochs):
            indexs = np.random.choice(range(len(X)), 10)
            batch_x = X[indexs]
            batch_y = Y[indexs]
            ls, _ = sess.run(
                [loss_mean, optimizer], feed_dict={train_x: batch_x, train_y: batch_y}
            )
            saver.save(sess, savedir + '/checkpoint', global_step=step)
            print(ls)


if __name__ == '__main__':
    train()
