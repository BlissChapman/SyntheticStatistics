import numpy as np
import torch

def uniform_noise(sample_length, batch_size, cuda):
    uniform_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sample_length))
    uniform_data = torch.Tensor(uniform_data)

    if cuda:
        uniform_data = uniform_data.cuda()

    return uniform_data

def gaussian(dataset_length):
    return np.random.normal(loc=0, scale=1.0, size=(dataset_length))

def gaussian_mixture(dataset_length):
    samples = np.random.normal(-2.5, 1.0, int(dataset_length/3))
    samples = np.append(samples, np.random.normal(-0.5, 0.35, int(dataset_length/3)))
    samples = np.append(samples, np.random.normal(1.0, 0.65, int(dataset_length/3)))
    return samples

def exponential(dataset_length):
    return np.random.exponential(scale=9.0, size=(dataset_length))

def chi_square(dataset_length):
    return np.random.chisquare(9, size=dataset_length)

def single_value(dataset_length):
    return 5*np.ones((dataset_length))
