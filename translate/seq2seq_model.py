# -*- coding:utf-8 -*-
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import datautil as data_util


class Seq2SeqModel(object):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        buckets,
        size,
        num_layers,
        dropout_keep_prob,
        max_gradient_norm,
        batch_size,
        learing_rate,
        learing_rate_decay_factor,
        use_lstm=False,
        num_samples=512,
        forward_only=False,
        dtype=tf.float32,
    ):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.dropout_keep_prob_output = dropout_keep_prob
        self.dropout_keep_prob_input = dropout_keep_prob
        self.learning_rate = tf.Variable(
            float(learing_rate), trainable=False, dtype=dtype
        )
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learing_rate_decay_factor
        )
        self.global_step = tf.Variable(0, trainable=False)
        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable('proj_w', [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b', [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sample_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(w, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size,
                    ),
                    dtype,
                )

            softmax_loss_function = sample_loss

            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                with tf.variable_scope('GRU') as scope:
                    cell = tf.contrib.rnn.DropoutWrapper(
                        tf.contrib.rnn.GRUCell(size),
                        input_keep_prob=self.dropout_keep_prob_input,
                        output_keep_prob=self.dropout_keep_prob_output,
                    )
                    if num_layers > 1:
                        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
                return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode,
                    dtype=dtype,
                )

            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(buckets[-1][0]):
                self.encoder_inputs.append(
                    tf.placeholder(tf.int32, shape=[None], name=f'encoder{i}')
                )
            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(
                    tf.placeholder(tf.int32, shape=[None], name=f'decoder{i}')
                )
            targets = [
                self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)
            ]

