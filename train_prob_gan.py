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
parser.add_argument('pth_to_train_data', help='the path to the data used to train the GAN')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# ========== OUTPUT DIRECTORIES ==========
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# ========== Hyperparameters ==========
NUM_TRAINING_STEPS = 50000
BATCH_SIZE = 8
MODEL_DIMENSIONALITY = 64
NOISE_SAMPLE_LENGTH = 64
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
LAMBDA = 1

description_f = open(args.output_dir + 'description.txt', 'w')
description_f.write('NUM_TRAINING_STEPS: {0}\n'.format(NUM_TRAINING_STEPS))
description_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
description_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
description_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
description_f.write('NOISE_SAMPLE_LENGTH: {0}\n'.format(NOISE_SAMPLE_LENGTH))
description_f.write('CRITIC_UPDATES_PER_GENERATOR_UPDATE: {0}\n'.format(CRITIC_UPDATES_PER_GENERATOR_UPDATE))
description_f.write('LAMBDA: {0}\n'.format(LAMBDA))
description_f.close()

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()
np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== Data ==========
def batch_generator(data, output_width, batch_size, cuda):
    epoch_length = len(data)

    while True:
        # Shuffle data between epochs:
        np.random.shuffle(real_data)

        for i in range(0, epoch_length, batch_size):
            # Retrieve data batch
            data_batch = np.array(data[i:i + batch_size])
            if len(data_batch) != batch_size:
                continue
            data_batch = data_batch.reshape((batch_size, output_width))

            # Create torch tensors
            data_batch = torch.Tensor(data_batch)

            if cuda:
                data_batch = data_batch.cuda()

            yield data_batch


real_data = np.load(args.pth_to_train_data)
output_width = real_data.shape[1] if len(real_data.shape) >= 2 else 1
real_data_generator = batch_generator(real_data, output_width, BATCH_SIZE, CUDA)

# ========== Models ==========
generator = models.ProbabilityDistGAN.Generator(input_width=NOISE_SAMPLE_LENGTH,
                                                output_width=output_width,
                                                dimensionality=MODEL_DIMENSIONALITY,
                                                cudaEnabled=CUDA)
critic = models.ProbabilityDistGAN.Critic(input_width=output_width,
                                          dimensionality=MODEL_DIMENSIONALITY,
                                          cudaEnabled=CUDA)

# ========= Training =========
for training_step in range(0, NUM_TRAINING_STEPS):
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
