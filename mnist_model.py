# -*- coding:utf-8 -*-
import tensorflow as tf
from absl import app as absl_app
from absl import flags


LEARNING_RATE = 1e-4


def create_model(data_format):
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]
    l = tf.keras.layers
    max_pool = l.MaxPool2D((2, 2), (2, 2), padding='same')
    return tf.keras.Sequential(
        [
            l.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
            l.Conv2D(
                32, 5, padding='same', data_format=data_format, activation=tf.nn.relu
            ),
            max_pool,
            l.Conv2D(
                64, 5, padding='same', data_format=data_format, activation=tf.nn.relu
            ),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dense(10),
        ]
    )


def model_fn(features, labels, mode, params):
    model = create_model(params['data_format'])
    image = features
    if isinstance(image, dict):
        image = features['image']
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(image, training=False)
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={'classify': tf.estimator.export.PredictOutput},
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=LEARNING_RATE)
        if params.get('multi_gpu'):
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        logits = model(image, training=True)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1)
        )
        tf.identity(LEARNING_RATE, 'learing_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()),
        )
    if mode == tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)
                )
            },
        )


def run_mnist(flags_obj):
    # model_helpers.apply_clean(flags_obj)
    model_function = model_fn
    num_gpus = flags_core.get_num_gpus(flags_obj)
    multi_gpu = num_gpus > 1
    if multi_gpu:
        distribution_utils.per_device_batch_size(flags_obj.batch_size, num_gpus)
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_fn,
            loss_reduction=tf.losses.Reduction.MEAN,
            devices=['/device:GPU:%d' % d for d in range(num_gpus)],
        )
    data_format = flags_obj.data_format
    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
        )
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=flags_obj.model_dir,
        params={'data_format': data_format, 'multi_gpu': multi_gpu},
    )

    def train_input_fn():
        ds = dataset.train(flags_obj.data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(flags_obj.batch_size)
        ds = ds.repeat(flags_obj.epochs_between_evals)
        return ds

    def eval_input_fn():
        return (
            dataset.test(flags_obj.data_dir)
            .batch(flags_obj.batch_size)
            .make_one_shot_iterator()
            .get_next()
        )

    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks, model_dir=flags_obj.model_dir, batch_size=flags_obj.batch_size
    )

    for _ in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
        mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(f'\nEvaluate results:\n\t{eval_results}\n')
        if model_helpers.past_stop_threshold(
            flags_obj.stop_threshold, eval_results['accuracy']
        ):
            break

    if flags_obj.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            {'image': image}
        )
        mnist_classifier.export_savedmodel(flags_obj.export_dir, input_fn)


def main(_):
    run_mnist(flags.FLAGS)


def define_mnist_flags():
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_mnist_flags()
    absl_app.run(main)
