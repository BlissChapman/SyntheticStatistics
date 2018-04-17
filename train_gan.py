import argparse
import datetime
import models.ProbabilityDistGAN
import numpy as np
import os
import shutil
import torch

from utils.sampling import sample, uniform_noise
from utils.plot import Plot
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="Train ProbabilityDistGAN")
parser.add_argument('distribution', choices=['gaussian_0', 'gaussian_0_1', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture'], help='the univariate probaility distribution from which training data should be sampled')
parser.add_argument('num_training_samples', type=int, help='the number of samples to use when training the ProbabilityDistGAN')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# ========== OUTPUT DIRECTORIES ==========
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# ========== Hyperparameters ==========
TRAINING_STEPS = 10000
BATCH_SIZE = 8
MODEL_DIMENSIONALITY = 64
SAMPLE_LENGTH = 1
NOISE_SAMPLE_LENGTH = 64
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 1
LAMBDA = 10
VISUALIZATION_INTERVAL = 10000

description_f = open(args.output_dir + 'description.txt', 'w')
description_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
description_f.write('TRAINING_STEPS: {0}\n'.format(TRAINING_STEPS))
description_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
description_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
description_f.write('SAMPLE_LENGTH: {0}\n'.format(SAMPLE_LENGTH))
description_f.write('NOISE_SAMPLE_LENGTH: {0}\n'.format(NOISE_SAMPLE_LENGTH))
description_f.write('CRITIC_UPDATES_PER_GENERATOR_UPDATE: {0}\n'.format(CRITIC_UPDATES_PER_GENERATOR_UPDATE))
description_f.write('LAMBDA: {0}\n'.format(LAMBDA))
description_f.write('VISUALIZATION_INTERVAL: {0}\n'.format(VISUALIZATION_INTERVAL))
description_f.close()

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()
np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== Data ==========
def batch_generator(data, sample_length, batch_size, cuda):
    epoch_length = len(data)

    while True:
        # Shuffle data between epochs:
        np.random.shuffle(real_data)

        for i in range(0, epoch_length, sample_length * batch_size):
            # Retrieve data batch
            data_batch_len = sample_length * batch_size
            data_batch = np.array(data[i:i + data_batch_len])
            if len(data_batch) != data_batch_len:
                continue
            data_batch = data_batch.reshape((batch_size, sample_length))

            # Create torch tensors
            data_batch = torch.Tensor(data_batch)

            if cuda:
                data_batch = data_batch.cuda()

            yield data_batch


real_data = sample(args.num_training_samples, distribution=args.distribution)
real_data_generator = batch_generator(real_data, SAMPLE_LENGTH, BATCH_SIZE, CUDA)

# ========== Models ==========
generator = models.ProbabilityDistGAN.Generator(input_width=NOISE_SAMPLE_LENGTH,
                                                output_width=SAMPLE_LENGTH,
                                                dimensionality=MODEL_DIMENSIONALITY,
                                                cudaEnabled=CUDA)
critic = models.ProbabilityDistGAN.Critic(input_width=SAMPLE_LENGTH,
                                          dimensionality=MODEL_DIMENSIONALITY,
                                          cudaEnabled=CUDA)

# ========= Training =========
for training_step in range(0, TRAINING_STEPS):
    # Train critic
    for critic_step in range(CRITIC_UPDATES_PER_GENERATOR_UPDATE):
        real_data_batch = Variable(next(real_data_generator))
        critic_noise_sample = Variable(uniform_noise(NOISE_SAMPLE_LENGTH, BATCH_SIZE, CUDA))
        synthetic_data_batch = generator(critic_noise_sample)
        _ = critic.train(real_data_batch, synthetic_data_batch, LAMBDA)

    generator_noise_sample = Variable(uniform_noise(NOISE_SAMPLE_LENGTH, BATCH_SIZE, CUDA))
    synthetic_data_batch = generator(generator_noise_sample)
    critic_output = critic(synthetic_data_batch)
    _ = generator.train(critic_output)

# Save models
torch.save(generator.state_dict(), "{0}generator".format(args.output_dir))
torch.save(critic.state_dict(), "{0}critic".format(args.output_dir))
