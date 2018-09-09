# -*- coding:utf-8 -*-
import os
import math
import sys
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import datautil
import seq2seq_model

tf.reset_default_graph()
steps_per_checkpoint = 200

max_train_data_size = 0

dropout = 0.9
grad_clip = 5.0
batch_size = 60

num_layers = 2
learning_rate = 0.5
lr_decay_factor = 0.99

hidden_size = 100
checkpoint_dir = 'fanyichina/checkpoints'

_buckets = [(20, 20), (40, 40), (50, 50), (60, 60)]


def getfanyi():
    vocaben, rev_vocaben = datautil.initialize_vocabulary(
        os.path.join(datautil.data_dir, datautil.vocabulary_fileen))
    vocab_sizeen = len(vocaben)

    vocabch, rev_vocabch = datautil.initialize_vocabulary(
        os.path.join(datautil.data_dir, datautil.vocabulary_filech))
    vocab_sizech = len(vocabch)

    filesfrom, _ = datautil.getRawFileList(datautil.data_dir+'fromids')
    filesto, _ = datautil.getRawFileList(datautil.data_dir+'toids/')
    source_train_file_path = filesfrom[0]
    target_train_file_path = filesto[0]
    return vocab_sizeen, vocab_sizech, rev_vocaben, rev_vocabch, source_train_file_path, target_train_file_path


def createModel(session, forward_only, from_vocab_size, to_vocab_size):
    model = seq2seq_model.Seq2SeqModel(
        from_vocab_size,
        to_vocab_size,
        _buckets,
        hidden_size,
        num_layers,
        dropout,
        grad_clip,
        batch_size,
        learning_rate,
        lr_decay_factor,
        forward_only=forward_only,
        dtype=tf.float32
    )
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt != None:
        model.saver.restore(session, ckpt)
    else:
        session.run(tf.global_variables_initializer())
    return model


def readData(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode='r') as source_file:
        with tf.gfile.GFile(source_path, mode='r') as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print(f'reading data line {counter}')
                sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target.readline()
    return data_set


def main():
    vocab_sizeen, vocab_sizech, rev_vocaben, rev_vocabch, source_train_file_path, target_train_file_path = getfanyi()
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with tf.Session() as sess:
        model = createModel(sess, False, vocab_sizeen, vocab_sizech)

        source_test_file_path = source_train_file_path
        target_test_file_path = target_train_file_path

        train_set = readData(source_train_file_path,
                             target_train_file_path, max_train_data_size)
        test_set = readData(source_test_file_path,
                            target_test_file_path, max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(
            train_bucket_sizes[:i+1])/train_total_size for i in xrange(len(train_bucket_sizes))]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(
                len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(
                sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            step_time += (time.time()-start_time)/steps_per_checkpoint
            loss += step_loss/steps_per_checkpoint
            current_step += 1
            if current_step % steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
            checkpoint_path = os.path.join(checkpoint_dir, 'seq2seqtest.ckpt')
            model.saver.save(sess, checkpoint_path,
                             global_step=model.global_step)
            step_time = 0.0, 0.0
            if len(test_set[bucket_id]) == 0:
                continue
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                test_set, bucket_id)
            _, eval_loss, output_logits = model.step(
                sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            inputstr = datautil.ids2texts(
                reversed([en[0] for en in encoder_inputs]), rev_vocaben)
            outputs = [np.argmax(logit, axis=1)[0] for logit in output_logits]
            if datautil.EOS_ID in outputs:
                outputs = outputs[:outputs.index(datautil.EOS_ID)]
        sys.stdout.flush()


if __name__ == '__main__':
    main()
