import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom

from params import Params, set_params


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images, labels, output_file):
    """Converts a file to TFRecords."""
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        num_entries_in_batch = len(labels)
        for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(images[i].tobytes()),
                    'label': _int64_feature(labels[i])
                }))
            record_writer.write(example.SerializeToString())


def load_data(num_valids=5000):
    dataset = Params.dataset
    data_path = os.path.join(Params.base_dir, 'data', dataset)
    if dataset == 'cifar10':
        images, labels = cifar10(data_path)
    elif dataset == 'svhn':
        images, labels = svhn(data_path)
    elif dataset == 'mnist':
        images, labels = mnist()
    elif dataset == 'fashion':
        images, labels = fashion()
    elif dataset == 'stl':
        images, labels = stl(data_path)
        num_valids = 500
    else:
        images, labels = {}, {}

    if num_valids:
        images["valid"] = images["train"][-num_valids:]
        labels["valid"] = labels["train"][-num_valids:]

        images["train"] = images["train"][:-num_valids]
        labels["train"] = labels["train"][:-num_valids]
    else:
        images["valid"], labels["valid"] = None, None

    tf.logging.info("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    tf.logging.info("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    tf.logging.info("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    if num_valids:
        images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std

    print(images['train'].shape)
    print(labels['train'].shape)
    print(images['valid'].shape)
    print(labels['valid'].shape)
    print(images['test'].shape)
    print(labels['test'].shape)
    print()
    save_path = os.path.join(Params.base_dir, 'tf_record_data')
    convert_to_tfrecord(images['train'], labels['train'], os.path.join(save_path, dataset, 'train.tfrecord'))
    convert_to_tfrecord(images['valid'], labels['valid'], os.path.join(save_path, dataset, 'valid.tfrecord'))
    convert_to_tfrecord(images['test'], labels['test'], os.path.join(save_path, dataset, 'test.tfrecord'))
    return images, labels


def mnist():
    mnist = tf.keras.datasets.mnist
    images, labels = {}, {}
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'minimum')
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
    x_test = np.concatenate((x_test, x_test, x_test), axis=-1)
    images['train'] = x_train.astype(np.float32)
    images['test'] = x_test.astype(np.float32)
    labels['train'] = y_train.flatten().astype(np.int32)
    labels['test'] = y_test.flatten().astype(np.int32)
    return images, labels


def fashion():
    mnist = tf.keras.datasets.fashion_mnist
    images, labels = {}, {}
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'minimum')
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
    x_test = np.concatenate((x_test, x_test, x_test), axis=-1)
    images['train'] = x_train.astype(np.float32)
    images['test'] = x_test.astype(np.float32)
    labels['train'] = y_train.flatten().astype(np.int32)
    labels['test'] = y_test.flatten().astype(np.int32)
    return images, labels


def resize_image_data(data, resize_shape):
    """Resize images to given dimension.
    Args:
        data: 1-D, 2-D or 3-D images. The Images are expected to have channel last configuration.
        resize_shape: Image resize dimension.
    Returns:
        data: Reshaped data.
    """
    if data is None or len(resize_shape) == 0:
        return data

    if len(data.shape) > 1 and np.array_equal(data[0].shape, resize_shape):
        return data

    output_data = []
    for im in data:
        output_data.append(zoom(input=im, zoom=np.divide(resize_shape, im.shape)))

    return np.array(output_data)


def stl(datapath):
    from scipy.io import loadmat

    def load_data(path):
        """ Helper function for loading a MAT-File"""
        full_name = os.path.join(datapath, path)
        data = loadmat(full_name)
        return data['X'], data['y']

    x_train, y_train = load_data('train.mat')
    x_test, y_test = load_data('test.mat')
    x_train = x_train.reshape((len(x_train), 96, 96, 3))
    x_test = x_test.reshape((len(x_test), 96, 96, 3))
    x_train = resize_image_data(x_train, (32.0, 32.0, 3.0))
    x_test = resize_image_data(x_test, (32.0, 32.0, 3.0))
    images, labels = {}, {}
    images['train'] = x_train.astype(np.float32)
    images['test'] = x_test.astype(np.float32)
    labels['train'] = y_train.flatten().astype(np.int32)
    labels['test'] = y_test.flatten().astype(np.int32)
    return images, labels


def svhn(datapath):
    from scipy.io import loadmat

    def load_data(path):
        """ Helper function for loading a MAT-File"""
        full_name = os.path.join(datapath, path)
        data = loadmat(full_name)
        return data['X'], data['y']

    x_train, y_train = load_data('train_32x32.mat')
    x_test, y_test = load_data('test_32x32.mat')
    x_train, y_train = x_train.transpose((3, 0, 1, 2)), y_train[:, 0]
    x_test, y_test = x_test.transpose((3, 0, 1, 2)), y_test[:, 0]
    images, labels = {}, {}
    images['train'] = x_train.astype(np.float32)
    images['test'] = x_test.astype(np.float32)
    labels['train'] = y_train.flatten().astype(np.int32)
    labels['test'] = y_test.flatten().astype(np.int32)
    return images, labels


def cifar10(data_path):
    def _read_data(data_path, train_files):
        """Reads CIFAR-10 format data. Always returns NHWC format.

      Returns:
        images: np tensor of size [N, H, W, C]
        labels: np tensor of size [N]
      """
        images, labels = [], []
        for file_name in train_files:
            tf.logging.info(file_name)
            full_name = os.path.join(data_path, file_name)
            with open(full_name, 'rb') as finp:
                data = pickle.load(finp, encoding='bytes')
                # data = pickle.load(finp)
                batch_images = data[b"data"].astype(np.float32) / 255.0
                batch_labels = np.array(data[b"labels"], dtype=np.int32)
                images.append(batch_images)
                labels.append(batch_labels)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        images = np.reshape(images, [-1, 3, 32, 32])
        images = np.transpose(images, [0, 2, 3, 1])

        return images, labels

    tf.logging.info("-" * 80)
    tf.logging.info("Reading data")

    images, labels = {}, {}

    train_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_file = [
        "test_batch",
    ]
    images["train"], labels["train"] = _read_data(data_path, train_files)
    images["test"], labels["test"] = _read_data(data_path, test_file)

    return images, labels


def _parse(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.float32)
    image.set_shape([3 * 32 * 32])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.reshape(image, [32, 32, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    image = tf.transpose(image, [2, 0, 1])

    return image, label


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    if initializer is None:
        initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def global_avg_pool(x, data_format="NHWC"):
    if data_format == "NHWC":
        x = tf.reduce_mean(x, [1, 2])
    elif data_format == "NCHW":
        x = tf.reduce_mean(x, [2, 3])
    else:
        raise NotImplementedError("Unknown data_format {}".format(data_format))
    return x


def read_data():
    path = os.path.join(Params.base_dir, 'tf_record_data', Params.dataset)
    train_data = tf.data.TFRecordDataset(os.path.join(path, 'train.tfrecord')).repeat().map(_parse)
    x_train, y_train = train_data.shuffle(buffer_size=45000) \
        .batch(160, drop_remainder=True).make_one_shot_iterator().get_next()
    w = create_weight("w", [3, 3, 3, 24 * 3])
    x = tf.nn.conv2d(
        x_train, w, [1, 1, 1, 1], "SAME", data_format="NCHW")
    x = global_avg_pool(x, data_format="NCHW")
    w = create_weight("w1", [24 * 3, 10])
    logits = tf.matmul(x, w)
    train_preds = tf.argmax(logits, axis=1)
    train_preds = tf.to_int32(train_preds)
    train_acc = tf.equal(train_preds, y_train)
    train_acc = tf.to_int32(train_acc)
    train_acc = tf.reduce_sum(train_acc)
    with tf.train.SingularMonitoredSession() as sess:
        a = sess.run([train_acc])
        print(a)


def main():
    set_params()
    read_data()
    # Params.dataset = 'cifar10'
    # load_data()
    # Params.dataset = 'mnist'
    # load_data()
    # Params.dataset = 'fashion'
    # load_data()
    # Params.dataset = 'stl'
    # load_data()
    # Params.dataset = 'svhn'
    # load_data()


if __name__ == '__main__':
    main()
