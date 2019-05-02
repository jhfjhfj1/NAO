import pickle

import numpy as np
import os
from scipy.special import softmax
from sklearn.metrics import pairwise_distances

from .data_utils import load_data, convert_to_tfrecord
from .params import Params, set_params
from .controller import encode, predict
from .utils import generate_arch, parse_arch_to_seq


def maximum_likelihood_estimation(samples, arch_similarity_matrix):
    # samples should be a tensor of (n, p, k)
    # arch_similarity_matrix should be a matrix of (p, p)
    x = samples
    m = np.mean(samples, -1)
    v = arch_similarity_matrix
    inverse_v = np.linalg.inv(v)

    k = x.shape[-1]
    n = m.shape[0]
    p = m.shape[1]

    u = np.zeros((n, n))
    for i in range(k):
        temp = x[:, :, i] - m
        u += np.matmul(np.matmul(temp, inverse_v), np.transpose(temp))
    u = u / (k * p)
    return u


def load_history():
    history = []
    controller_paths = []
    for file in os.listdir(os.fsencode(Params.history_dir)):
        dir_path = os.path.join(os.fsencode(Params.history_dir), os.fsencode(file), os.fsencode('child'))
        controller_paths.append(os.path.join(os.fsencode(Params.history_dir), os.fsencode(file), os.fsencode('controller')))
        if not os.path.isdir(dir_path):
            continue
        arch_list = []
        perf_list = []
        for my_file in os.listdir(dir_path):
            if not my_file.decode("utf-8").startswith('history'):
                continue
            path = os.path.join(dir_path, my_file)
            archs, perf = pickle.load(open(path, 'rb'))
            branch_length = Params.source_length // 2 // 5 // 2
            encoder_input = list(map(lambda x:
                                     parse_arch_to_seq(x[0], branch_length) + parse_arch_to_seq(x[1], branch_length),
                                     archs))
            arch_list += encoder_input
            perf_list += perf
        history.append((arch_list, perf_list))
    return history, controller_paths


def augment(encoder_input, predictor_output):
    history, controller_paths = load_history()
    history.append((encoder_input, predictor_output))
    arch_to_id = {}
    arch_cnt = 0
    for arch_list, perf_list in history:
        for arch in arch_list:
            hashable_arch = tuple(arch)
            if hashable_arch not in arch_to_id:
                arch_to_id[hashable_arch] = arch_cnt
                arch_cnt += 1
    unique_archs = [list(arch) for arch in sorted(arch_to_id.keys(), key=lambda x: arch_to_id[x])]
    arch_embeddings, _ = encode(unique_archs)

    n_tasks = len(history)
    n_archs = len(unique_archs)
    mean = np.zeros((n_tasks, n_archs))
    for index, (arch_list, perf_list) in enumerate(history):
        mean[index] = np.array(predict(unique_archs)).flatten()
        for arch, perf in zip(arch_list, perf_list):
            mean[index, arch_to_id[tuple(arch)]] = perf

    # similarity_matrix = np.matmul(embeddings, np.transpose(embeddings))
    task_embeddings = np.matmul(mean, np.linalg.pinv(np.transpose(arch_embeddings)))
    task_similarity_matrix = pairwise_distances(task_embeddings, metric="cosine")
    similarity_key = softmax(task_similarity_matrix[-1][:-1])
    task_ids = sorted(range(n_tasks - 1), key=lambda x: similarity_key[x])

    ret_history = []
    for task_id, key in zip(task_ids, similarity_key):
        ret_history.append((history[task_id][0], history[task_id][1], key))

    return ret_history


def synthesize_history():
    dir_list = ['a', 'b', 'c', 'd']
    for current_dir in dir_list:
        path = os.path.join(Params.history_dir, current_dir)
        if not os.path.exists(path):
            os.mkdir(path)
        archs = generate_arch(Params.num_seed_arch, Params.num_cells, 5)  # [[[conv],[reduc]]]
        perfs = np.random.rand(len(archs))
        for epoch in range(10):
            pickle.dump((archs, perfs), open(os.path.join(path, 'history.{}'.format(epoch)), 'wb'))


def filter_classes(images, labels, classes):
    ret_images = []
    ret_labels = []
    for index, (image, label) in enumerate(zip(images, labels)):
        if label in classes:
            ret_images.append(image)
            ret_labels.append(label)
    return np.array(ret_images), np.array(ret_labels)


def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def slice_dataset_by_class(images, labels, save_path):
    for start in range(10):
        classes = [start, start + 1, start + 2]
        images['train'], labels['train'] = filter_classes(images['train'], labels['train'], classes)
        images['valid'], labels['valid'] = filter_classes(images['valid'], labels['valid'], classes)
        images['test'], labels['test'] = filter_classes(images['test'], labels['test'], classes)
        path = os.path.join(save_path, '{}_{}_{}'.format(start, start + 1, start + 2))
        ensure_dir(path)
        convert_to_tfrecord(images['train'], labels['train'], os.path.join(path, 'train.tfrecord'))
        convert_to_tfrecord(images['valid'], labels['valid'], os.path.join(path, 'valid.tfrecord'))
        convert_to_tfrecord(images['test'], labels['test'], os.path.join(path, 'test.tfrecord'))
        # pickle.dump((images, labels), open(path, 'wb'))


def slice_datasets():
    Params.dataset = 'mnist'
    images, labels = load_data()
    slice_dataset_by_class(images, labels, 'tf_record_data/sliced_mnist')
    Params.dataset = 'cifar10'
    images, labels = load_data()
    slice_dataset_by_class(images, labels, 'tf_record_data/sliced_cifar10')


def main():
    set_params()
    # n = 10
    # p = 20
    # k = 30
    # embedding_len = 40
    # samples = np.random.rand(n, p, k)
    # arch_embeds = np.random.rand(p, embedding_len)
    # arch_similarity_matrix = np.matmul(arch_embeds, np.transpose(arch_embeds))
    # print(maximum_likelihood_estimation(samples, arch_similarity_matrix).shape)

    slice_datasets()

    # synthesize_history()

    # old_archs = utils.generate_arch(Params.num_seed_arch, Params.num_cells, 5)
    # branch_length = Params.source_length // 2 // 5 // 2
    # encoder_input = list(map(lambda x:
    #                          utils.parse_arch_to_seq(x[0], branch_length) + utils.parse_arch_to_seq(x[1],
    #                                                                                                 branch_length),
    #                          old_archs))
    # predictor_target = np.random.rand(len(encoder_input))
    # history = augment(encoder_input, predictor_target)
    # print(history)


if __name__ == '__main__':
    main()
