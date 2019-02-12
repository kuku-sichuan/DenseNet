import os
import tensorflow as tf
from multiprocessing import cpu_count


def train_parse_fn(serialized_example, all_config):

    ################################################
    # the reading process
    ################################################
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([all_config.CHANNELS * all_config.IMAGE_SIZE * all_config.IMAGE_SIZE])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [all_config.CHANNELS, all_config.IMAGE_SIZE, all_config.IMAGE_SIZE]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    ####################################################
    # image preprocessing
    ####################################################
    # sub means
    image -= tf.convert_to_tensor(all_config.IMAGE_MEANS, tf.float32)
    # div stds
    image /= tf.convert_to_tensor(all_config.IMAGE_STD, tf.float32)
    # Pad 4 pixels on each dimension of feature map, done in mini-batch
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [all_config.IMAGE_SIZE, all_config.IMAGE_SIZE, all_config.CHANNELS])
    image = tf.image.random_flip_left_right(image)
    if all_config.DATA_FORMAT == "channels_first":
        image = tf.transpose(image, [2, 0, 1])
    return image, label


def train_input_fn(all_config):

    dataset = tf.data.TFRecordDataset(os.path.join(os.getcwd(), "data/train.tfrecords"))
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=all_config.NUM_GPU*all_config.BATCH_SIZE,
                                                                    count=all_config.EPOCH))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda x: train_parse_fn(x, all_config),
                                                               all_config.BATCH_SIZE,
                                                               num_parallel_batches=cpu_count() // 2))
    dataset = dataset.prefetch(all_config.BATCH_SIZE*all_config.NUM_GPU)
    return dataset


def test_parse_fn(serialized_example, all_config):

    ################################################
    # the reading process
    ################################################
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([all_config.CHANNELS * all_config.IMAGE_SIZE * all_config.IMAGE_SIZE])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [all_config.CHANNELS, all_config.IMAGE_SIZE, all_config.IMAGE_SIZE]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    # sub means
    image -= tf.convert_to_tensor(all_config.IMAGE_MEANS, tf.float32)
    # div stds
    image /= tf.convert_to_tensor(all_config.IMAGE_STD, tf.float32)
    if all_config.DATA_FORMAT == "channels_first":
        image = tf.transpose(image, [2, 0, 1])
    return image, label


def test_input_fn(all_config):

    dataset = tf.data.TFRecordDataset(os.path.join(os.getcwd(), "data/eval.tfrecords"))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda x: test_parse_fn(x, all_config),
                                                               all_config.BATCH_SIZE,
                                                               num_parallel_batches=cpu_count() // 2))
    dataset = dataset.prefetch(all_config.BATCH_SIZE*all_config.NUM_GPU)

    return dataset
