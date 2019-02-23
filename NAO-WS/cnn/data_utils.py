import os
import sys
import pickle
import numpy as np
import tensorflow as tf


def read_data(data_path, num_valids=5000, dataset='CIFAR10'):
    if dataset == 'CIFAR10':
        images, labels = cifar10(data_path)
    elif dataset == 'SVHN':
        images, labels = svhn(data_path)
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

    return images, labels


def mnist():
    mnist = tf.keras.datasets.mnist
    images, labels = {}, {}
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images['train'] = x_train
    images['test'] = x_test
    labels['train'] = y_train.flatten()
    labels['test'] = y_test.flatten()
    return images, labels


def fashion():
    mnist = tf.keras.datasets.fashion_mnist
    images, labels = {}, {}
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images['train'] = x_train
    images['test'] = x_test
    labels['train'] = y_train.flatten()
    labels['test'] = y_test.flatten()
    return images, labels


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
    images, labels = {}, {}
    images['train'] = x_train
    images['test'] = x_test
    labels['train'] = y_train.flatten()
    labels['test'] = y_test.flatten()
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
    images['train'] = x_train
    images['test'] = x_test
    labels['train'] = y_train.flatten()
    labels['test'] = y_test.flatten()
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
