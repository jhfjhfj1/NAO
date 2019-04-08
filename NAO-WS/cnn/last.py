import numpy as np


def rescale(predicted, target):
    pass


def maximum_likelihood_estimation(samples, arch_similarity_matrix):
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
