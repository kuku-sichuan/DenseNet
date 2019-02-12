import os
import numpy as np
import tensorflow as tf
from config import Cifar10Config
from data_processing import train_input_fn, test_input_fn
from run_meta import MetadataHook
from model import DenseNet

# let the information log on the terminal
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features,
             labels,
             mode,
             params):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    all_config = params["config"]

    logits = DenseNet(features, all_config.GROWTH_RATE,
                                all_config.DEPTH,
                                all_config.NUM_DENSE_BLOCK,
                                all_config.NUM_INIT_FILTER,
                                all_config.SUB_SAMPLE_IMAGE,
                                all_config.NUM_CLASSES,
                                training=is_training,
                                bottleneck=all_config.BOTTLENECK,
                                dropout_rate=all_config.DROPOUT_RATES,
                                compression=all_config.COMPRESSION,
                                data_format=all_config.DATA_FORMAT,
                                all_config=all_config)

    with tf.variable_scope("loss"):
        classifier_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

        regularization_list = [tf.reduce_sum(all_config.WEIGHT_DECAY * tf.square(w.read_value()))
                               for w in tf.trainable_variables()]
        regularization_loss = tf.add_n(regularization_list)

        total_loss = classifier_loss + regularization_loss
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(all_config.BOUNDARY[0]), np.int64 (all_config.BOUNDARY[1])],
                                     values=[all_config.INIT_LEARNING_RATE, all_config.INIT_LEARNING_RATE / 10,
                                             all_config.INIT_LEARNING_RATE / 100])
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_op = optimizer.minimize(total_loss, global_step)

    predictions = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
    accuracies, update_accuracies = tf.metrics.accuracy(labels, predictions)

    meta_hook = MetadataHook(save_steps=all_config.SAVE_EVERY_N_STEP*all_config.EPOCH/2, output_dir=all_config.LOG_OUTPUT_DIR)
    summary_hook = tf.train.SummarySaverHook(save_steps=all_config.SAVE_EVERY_N_STEP,
                                             output_dir=os.path.join(all_config.LOG_OUTPUT_DIR, all_config.NET_NAME),
                                             summary_op=tf.summary.merge_all())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                          train_op=train_op,
                                          training_hooks=[meta_hook, summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                          eval_metric_ops={'accuracies': (accuracies, update_accuracies)})


if __name__=="__main__":
    all_config = Cifar10Config()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    session_config.allow_soft_placement = True
    estimator_config = tf.estimator.RunConfig(model_dir=os.path.join(all_config.LOG_OUTPUT_DIR, all_config.NET_NAME),
                                              log_step_count_steps=200,
                                              save_summary_steps=all_config.SAVE_EVERY_N_STEP,
                                              save_checkpoints_steps=all_config.SAVE_EVERY_N_STEP,
                                              session_config=session_config)
    densenet_estimator = tf.estimator.Estimator(model_fn,
                                               params={"config": all_config},
                                               config=estimator_config)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(all_config))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn(all_config),
                                      start_delay_secs=100,
                                      throttle_secs=120)
    tf.estimator.train_and_evaluate(densenet_estimator, train_spec, eval_spec)
