# -*- coding:utf-8 -*-
import os

import numpy as np
import tensorflow as tf

from translate import datautil
import seq2seq_model

_buckets = []
convo_hist_limit = 1
max_source_length = 1
max_target_length = 2

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.reset_default_graph

max_train_data_size = 0

data_dir = 'datacn/'

dropout = 1.0
grad_clip = 5.0
batch_size = 60
hidden_size = 14
num_layers = 2
learning_rate = 0.5
lr_decay_factor = 0.99

checkpoint_dir = 'data/checkpoints/'

hidden_size = 100
checkpoint_dir = 'fanyichina/checkpoints/'
data_dir = 'fanyichina'
_buckets = [(20, 20), (40, 40), (50, 50), (60, 60)]


def getfanyiInfo():
    vocaben, rev_vocaben = datautil.initialize_vocabulary(
        os.path.join(datautil.data_dir, datautil.vocabulary_fileen))
    vocab_sizeen = len(vocaben)
    vocabch, rev_vocabch = datautil.initialize_vocabulary(
        os.path.join(datautil.data_dir, datautil.vocabulary_filech))
    vocab_sizech = len(vocabch)
    return vocab_sizeen, vocab_sizech, vocaben, vocabch


def createModel(session, forward_only, from_vocab_size, to_vocab_size):
    model = seq2seq_model.Seq2SeqModel(from_vocab_size, to_vocab_size, _buckets, hidden_size, num_layers, dropout,
                                       grad_clip, batch_size, learning_rate, lr_decay_factor, forward_only=forward_only, dtype=tf.float32)
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt != None:
        model.saver.restore(session, ckpt)
    else:
        session.run(tf.global_variables_initializer())
    return model


def main():
    vocab_sizeen, vocab_sizech, vocaben, rev_vocabch = getfanyiInfo()
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with tf.Session() as sess:
        model = createModel(sess, True, vocab_sizeen, vocab_sizech)
        model.batch_size = 1
        conversation_history = []
        while True:
            prompt = '请输入:'
            sentence = input(prompt)
            conversation_history.append(sentence)
            conversation_history = conversation_history[-conversation_history:]

            token_ids = list(reversed(datautil.sentence_to_ids(
                " ".join(conversation_history), vocaben, normalize_digits=True, Isch=True)))
            bucket_id = min([b for b in range(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])

            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)
            _, _, output_logits = model.step(
                sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1))
                       for logit in output_logits]
            if datautil.EOS_ID in outputs:
                outputs = outputs[:outputs.index(datautil.EOS_ID)]
                convo_output = " ".join(
                    datautil.ids2texts(outputs, rev_vocabch))
                conversation_history.append(convo_output)
            else:
                print('can not translation!')


if __name__ == '__main__':
    main()
