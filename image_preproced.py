# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.gfile as gfile


def read_image(filename_queue):
    class ArticleRecord(object):
        pass
    result = ArticleRecord()
    label_bytes = 1
    result.height = 500
    result.width = 500
    result.depth = 3
    image_bytes = result.height*result.width*result.depth
    record_bytes = label_bytes+image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [
                             result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples+3*batch_size, min_after_dequeue=min_queue_examples)
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    filenames = ''
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_image(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = 500
    width = 500
    distorted_image = tf.image._random_crop(reshaped_image, [height, width])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_contrast(
        distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(500*min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size)
