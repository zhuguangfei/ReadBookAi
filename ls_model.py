# -*- coding:utf-8 -*-
import tensorflow as tf


def model_fn(features, labels, mode, params):
    def _get_input_tensors(features, labels):
        shapes = features['shape']
        lenghts = tf.squeeze(
            tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1])
        )
        inks = tf.reshape(features['ink'], [params.batch_size, -1, 3])
        if labels is not None:
            labels = tf.squeeze(labels)
        return inks, lenghts, labels

    def _add_conv_layers(inks, lengths):
        convolved = inks
        for i in range(len(params.num_conv)):
            convolved_input = convolved
            if params.batch_norm:
                convolved_input = tf.layers.batch_normalization(
                    convolved_input, training=(mode == tf.estimator.ModeKeys.TRAIN)
                )
            if i > 0 and params.dropout:
                convolved_input = tf.layers.dropout(
                    convolved_input,
                    rate=params.dropout,
                    training=(mode == tf.estimator.ModeKeys.TRAIN),
                )
            convolved = tf.layers.conv1d(
                convolved_input,
                filters=params.num_conv[i],
                kernel_size=params.conv_len[i],
                activation=None,
                strides=1,
                padding='same',
                name='conv1d_%d' % i,
            )
        return convolved, lenghts

    def _add_regular_rnn_layers(convolved, lenghts):
        if params.cell_type == 'lstm':
            cell = tf.nn.rnn_cell.BasicLSTMCell
        elif params == 'block_lstm':
            cell = tf.contrib.rnn.LSTMBlockCell
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        outputs, _, _ = tf.contrib.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=convolved,
            sequence_length=lenghts,
            dtype=tf.float32,
            scope='rnn_classification',
        )
        return outputs

    def _add_cudnn_rnn_layers(convolved):
        convolved = tf.transpose(convolved, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=params.num_layers,
            num_units=params.num_nodes,
            dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            direction='bidirectional',
        )
        outputs, _ = lstm(convolved)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs

    def _add_rnn_layers(convolved, lenghts):
        if params.cell_type != 'cudnn_lstm':
            outputs = _add_regular_rnn_layers(convolved, lenghts)
        else:
            outputs = _add_cudnn_rnn_layers(convolved)
        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(lenghts, tf.shape(outputs)[1]), 2),
            [1, 1, tf.shape(outputs)[2]],
        )
        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        outputs = tf.reduce_sum(zero_outside, axis=1)
        return outputs

    def _add_fc_layers(final_state):
        return tf.layers.dense(final_state, params.num_classes)

    inks, lenghts, labels = _get_input_tensors(features, labels)
    convolved, lenghts = _add_conv_layers(inks, lenghts)
    final_state = _add_rnn_layers(convolved, lenghts)
    logits = _add_fc_layers(final_state)
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer='Adam',
        clip_gradients=params.gradient_clipping_norm,
        summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
    )
    predictions = tf.argmax(logits, axis=1)
    return tf.estimator.Estimator(
        mode=mode,
        predictions={'logits': logits, 'predictions': predictions},
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={'accuracy': tf.metrics.accuracy(labels, predictions)},
    )


def create_estimator_and_specs(run_config):
    pass
