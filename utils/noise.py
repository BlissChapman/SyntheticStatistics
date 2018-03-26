import numpy as np
import torch

def uniform_noise(sample_length, batch_size, cuda):
    uniform_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, sample_length))
    uniform_data = torch.Tensor(uniform_data)

    if cuda:
        uniform_data = uniform_data.cuda()

    return uniform_data
