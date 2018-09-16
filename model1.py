# -*- coding:utf-8 -*-
import tensorflow as tf

from data_util import get_xy, get_xy_test
import numpy as np


def create_variable(name, shape, initializer):
    var = tf.get_variable(
        name=name, shape=shape, dtype=tf.float32, initializer=initializer
    )
    return var


def create_variable_with_weight_decay(name, shape, stddev, wd):
    var = create_variable(
        name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def build_graph(input_x, batch_size):
    with tf.variable_scope('conv1') as scope:
        kernel = create_variable_with_weight_decay(
            'weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0
        )
        conv = tf.nn.conv2d(input_x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = create_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'
    )

    norml1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 /
                       9.0, beta=0.75, name='norml1')

    with tf.variable_scope('conv2') as scope:
        kernel = create_variable_with_weight_decay(
            'weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0
        )
        conv = tf.nn.conv2d(norml1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = create_variable('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    norml2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 /
                       9.0, beta=0.75, name='norml2')
    pool2 = tf.nn.max_pool(
        norml2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'
    )

    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[:-1].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [dim, -1])
        weights = create_variable_with_weight_decay(
            'weights', shape=[reshape[1], 384], stddev=1e-4, wd=0.004
        )
        biases = create_variable('biases', [384], tf.constant_initializer(0.0))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                            biases, name=scope.name)

    with tf.variable_scope('local4') as scope:
        weights = create_variable_with_weight_decay(
            'weights', shape=[384, 192], stddev=1e-4, wd=0.004
        )
        biases = create_variable('biases', [192], tf.constant_initializer(0.0))
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                            biases, name=scope.name)
    with tf.variable_scope('softmax_linear') as scope:
        weights = create_variable_with_weight_decay(
            'weights', [192, words_len], stddev=1 / 192, wd=0.0
        )
        biases = create_variable(
            'biases', [words_len], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases)
    return softmax_linear


def loss(logits, labels, labels_weights):
    losses = []
    for k in range(words_len):
        logits_k = logits[k]
        label_k = labels[k]
        loss_k = tf.losses.sparse_softmax_cross_entropy(
            labels=label_k, logits=logits_k
        )
        losses.append(loss_k)
    loss_ = tf.reduce_mean(losses)

    predicts = []
    corrects = []
    # task_accuracys = []
    # for i in range(batch_size):
    #     predict = tf.cast(tf.argmax(logits[i], axis=1), 'int32')
    #     predict = tf.multiply(predict, tf.cast(labels_weights[i], 'int32'))
    #     # correct = tf.cast(tf.equal(predict, tf.unstack(labels, axis=0)[i]), 'float')
    #     # correct = tf.multiply(correct, labels_weights[i])
    #     # corrects.append(correct)
    #     # accuracy = tf.reduce_sum(correct) / tf.reduce_sum(labels_weights)
    #     predicts.append(predict)
    #     # task_accuracys.append(accuracy)

    return loss_, predicts, corrects


batch_size = 10
height = 250
weigth = 250
channel = 3
max_len = 35
words_len = 2022
learning_rate = 0.001
train_epochs = 30


def get_batch(X, Y):
    indexs = np.random.choice(len(X), batch_size)
    label_weights = []
    batch_x = X[indexs]
    batch_y = Y[indexs]
    Y = np.zeros((words_len, batch_size))
    for y, v in enumerate(batch_y):
        weight = []
        for k, yk in enumerate(v):
            Y[k, v] = yk
    return batch_x, Y


def train():
    X, Y = get_xy()
    train_x = tf.placeholder(tf.float32, [height, weigth, channel, None])
    train_y = tf.placeholder(tf.int32, [words_len, None])
    # train_y = tf.unstack(train_y, axis=0)
    # y_weights = tf.placeholder(tf.int32, [words_len, None])
    # y_weights = tf.unstack(y_weights, axis=0)
    softmax_linear = build_graph(train_x, batch_size)
    loss_, predicts, corrects = loss(softmax_linear, train_y)
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate=learning_rate).minimize(loss_)
    savedir = 'save'
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_epochs):
            batch_x, batch_y, label_weights = get_batch(X, Y)
            ls, _ = sess.run(
                [corrects, optimizer],
                feed_dict={
                    train_x: batch_x,
                    train_y: batch_y,
                    # y_weights: label_weights,
                },
            )
            saver.save(sess, savedir + '/checkpoint', global_step=step)
            print('-' * 100, ls)


def predict():
    X = get_xy_test()
    predict_x = tf.placeholder(tf.float32, [None, 250, 250, 3])
    softmax_linear = build_graph(predict_x, 1)
    # predict_y = tf.cast(tf.argmax(softmax_linear, axis=1), "int32")
    # predict = tf.multiply(predict, tf.cast(self.expand_label_weights[i],'int32'))
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('save')
        saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.global_variables_initializer())
        softmax_linear = sess.run([softmax_linear], {predict_x: X})
        print(softmax_linear)
        # with open('a.txt', 'w') as w:
        # for i in predict_y:
        # w.write(f'{i}\n')


if __name__ == '__main__':
    train()
    # predict()
