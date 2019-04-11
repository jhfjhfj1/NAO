import numpy as np
import os

from params import Params
from controller import encode


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
    for file in os.listdir(os.fsencode(Params.history_dir)):
        dir_name = os.fsdecode(file)
        if os.path.isdir(dir_name):
            arch_list = []
            perf_list = []
            for file in os.listdir(dir_name):
                file_name = os.fsdecode(file)
                path = os.path.join(dir_name, file_name)
                archs, perf = open(path, 'rb')
                arch_list += archs
                perf_list += perf
            history.append((arch_list, perf_list))
    return history


def augment(encoder_input, predictor_output):
    history = load_history()
    arch_to_id = {}
    arch_cnt = 0
    for arch_list, perf_list in history:
        for arch in arch_list:
            if arch not in arch_to_id:
                arch_to_id[arch] = arch_cnt
                arch_cnt += 1
    unique_archs = sorted(arch_to_id.keys(), key=lambda x: arch_to_id[x])
    embeddings = encode(unique_archs)

    similarity_matrix = np.matmul(embeddings, np.transpose(embeddings))

    # TODO: construct x. How to deal with the missing values?

    return encoder_input, predictor_output


def main():
    n = 10
    p = 20
    k = 30
    embedding_len = 40
    samples = np.random.rand(n, p, k)
    arch_embeds = np.random.rand(p, embedding_len)
    arch_similarity_matrix = np.matmul(arch_embeds, np.transpose(arch_embeds))
    print(maximum_likelihood_estimation(samples, arch_similarity_matrix).shape)


if __name__ == '__main__':
    main()
