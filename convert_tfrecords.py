import argparse
import os
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange,urllib
import tensorflow as tf

CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'
tf.app.flags.DEFINE_string('data_dir', 'data', 'Directory to download and extract CIFAR-10 to.')
FLAGS = tf.app.flags.FLAGS

def download_and_extract(data_dir):
    # download CIFAR-10 if not already downloaded.
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = CIFAR_DOWNLOAD_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % ('cifar10',
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(CIFAR_DOWNLOAD_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, CIFAR_LOCAL_FOLDER)
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
    file_names['eval'] = ['test_batch']
    return file_names


def read_pickle_from_file(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict


def convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))
                record_writer.write(example.SerializeToString())


def main(data_dir):
    print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
    download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)
    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file)
    print('Done!')


if __name__ == '__main__':
    main(FLAGS.data_dir)
