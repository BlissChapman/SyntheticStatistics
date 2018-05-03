import numpy as np
import torch


def sample(dataset_length, distribution):
    if distribution == 'gaussian_0':
        return gaussian_0(dataset_length)
    elif distribution == 'gaussian_0_1':
        return gaussian_0_1(dataset_length)
    elif distribution == 'gaussian_1':
        return gaussian_1(dataset_length)
    elif distribution == 'chi_square_9':
        return chi_square_9(dataset_length)
    elif distribution == 'exp_9':
        return exp_9(dataset_length)
    elif distribution == 'gaussian_mixture':
        return gaussian_mixture(dataset_length)
    elif distribution == 'm_gaussian_0_0':
        return m_gaussian_0_0(dataset_length)
    elif distribution == 'm_gaussian_1_1':
        return m_gaussian_1_1(dataset_length)
    else:
        raise ValueError('Attempted to sample from a distribution that is not supported.')


# ========== UNIVARIATE DISTRIBUTIONS ==========
def gaussian_0(dataset_length):
    return np.random.normal(loc=0, scale=1.0, size=(dataset_length))


def gaussian_0_1(dataset_length):
    return np.random.normal(loc=0.1, scale=1.0, size=(dataset_length))


def gaussian_1(dataset_length):
    return np.random.normal(loc=1.0, scale=1.0, size=(dataset_length))


def gaussian_mixture(dataset_length):
    samples = np.random.normal(-2.5, 1.0, int(dataset_length / 3))
    samples = np.append(samples, np.random.normal(-0.5, 0.35, int(dataset_length / 3)))
    samples = np.append(samples, np.random.normal(1.0, 0.65, int(dataset_length / 3)))
    return samples


def exp_9(dataset_length):
    return np.random.exponential(scale=9.0, size=(dataset_length))


def chi_square_9(dataset_length):
    return np.random.chisquare(9, size=dataset_length)

# ========== MULTIVARIATE DISTRIBUTIONS ==========
def m_gaussian_0_0(dataset_length):
    return np.random.multivariate_normal([0]*(10), np.identity((10)), size=(dataset_length))

def m_gaussian_1_1(dataset_length):
    return np.random.multivariate_normal([1.0]*(10), np.identity((10)), size=(dataset_length))


# ========== NOISE ==========
def uniform_noise(sample_length, batch_size, cuda):
    uniform_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sample_length))
    uniform_data = torch.Tensor(uniform_data)

    if cuda:
        uniform_data = uniform_data.cuda()

    return uniform_data

def noise(size, cuda=False):
    noise = torch.from_numpy(np.random.normal(0.0, size=size)).float()
    if cuda:
        noise = noise.cuda()
    return noise
