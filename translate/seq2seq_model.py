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
            w_t = tf.get_variable(
                'proj_w', [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable(
                'proj_b', [self.target_vocab_size], dtype=dtype)
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

            if forward_only:
                self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets, self.target_weights, buckets, lambda x, y: seq2seq_f(
                        x, y, True), softmax_loss_function=softmax_loss_function
                )
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [tf.matmul(
                            output, output_projection[0])+output_projection[1] for output in self.outputs[b]]
                else:
                    self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                        self.encoder_inputs, self.decoder_inputs, targets, self.target_weights,
                        buckets, lambda x, y: seq2seq_f(x, y, False),
                        softmax_loss_function=softmax_loss_function)
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(
                        gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(
                        opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
            self.saver = tf.train.Saver(tf.global_variables())

    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_pad = [data_util.PAD_ID]*(encoder_size-len(encoder_input))
            encoder_inputs.append(
                list(reversed(encoder_size-len(encoder_input))))
            decoder_pad_size = decoder_size-len(decoder_input)-1
            decoder_inputs.append(
                [data_util.GO_ID]+decoder_input+[data_util.PAD_ID]*decoder_pad_size)
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array(
                [encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array(
                [decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_weights = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size-1:
                    target = decoder_input[batch_idx][length_idx+1]
                if length_idx == decoder_size-1 or target == data_util.PAD_ID:
                    batch_weights[batch_idx] = 0.0
            batch_weights.append(batch_weights)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self,session,encoder_inputs,decoder_inputs,target_weights,bucket_id,forward_only):
        encoder_size,decoder_size=self.buckets[bucket_id]
        if len(encoder_inputs)!=encoder_size:
            raise ValueError()
        if len(decoder_inputs)!=decoder_size:
            raise ValueError()
        if len(targets_weigths)!=decoder_size:
            raise ValueError()

        input_feed={}
        
