import matplotlib
matplotlib.use('Agg')

import argparse
import datetime
import models.ProbabilityDistGAN
import numpy as np
import os
import shutil
import timeit
import torch
import utils.utils

from utils.plot import Plot
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="Train ProbabilityDistGAN")
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# ========== OUTPUT DIRECTORIES ==========
DATA_OUTPUT_DIR = args.output_dir + 'data/'
VIS_OUTPUT_DIR = args.output_dir + 'visualizations/'
MODEL_OUTPUT_DIR = args.output_dir + 'models/'

shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)
os.makedirs(DATA_OUTPUT_DIR)
os.makedirs(VIS_OUTPUT_DIR)
os.makedirs(MODEL_OUTPUT_DIR)

# ========== Hyperparameters ==========
TRAINING_STEPS = 25000
BATCH_SIZE = 50
MODEL_DIMENSIONALITY = 64
SAMPLE_LENGTH = 4096
NOISE_SAMPLE_LENGTH = 128
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 1
LAMBDA = 10
VISUALIZATION_INTERVAL = 1000

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
if CUDA:
    print("Using GPU optimizations!")

np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== Data ==========
def gaussian(batch_size, sample_length, cuda):
    while True:
        gaussian_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, sample_length))
        gaussian_data = torch.Tensor(gaussian_data)

        if cuda:
            gaussian_data = gaussian_data.cuda()

        yield gaussian_data

def uniform(batch_size, sample_length, cuda):
    while True:
        uniform_data = np.random.uniform(low=0.0, high=5.0, size=(batch_size, sample_length))
        uniform_data = torch.Tensor(uniform_data)

        if cuda:
            uniform_data = uniform_data.cuda()

        yield uniform_data

gaussian_generator = gaussian(BATCH_SIZE, SAMPLE_LENGTH, CUDA)
uniform_generator = uniform(BATCH_SIZE, SAMPLE_LENGTH, CUDA)

# ========== Models ==========
generator = models.ProbabilityDistGAN.Generator(input_width=NOISE_SAMPLE_LENGTH,
                                                output_width=SAMPLE_LENGTH,
                                                dimensionality=MODEL_DIMENSIONALITY,
                                                cudaEnabled=CUDA)
critic = models.ProbabilityDistGAN.Critic(input_width=SAMPLE_LENGTH,
                                          dimensionality=MODEL_DIMENSIONALITY,
                                          cudaEnabled=CUDA)

# ========= Training =========
critic_losses_per_vis_interval = []
generator_losses_per_vis_interval = []

running_critic_loss = 0.0
running_generator_loss = 0.0
running_batch_start_time = timeit.default_timer()

for training_step in range(1, TRAINING_STEPS + 1):
    print("BATCH: [{0}/{1}]\r".format(training_step % VISUALIZATION_INTERVAL, VISUALIZATION_INTERVAL), end='')

    # Train critic
    for critic_step in range(CRITIC_UPDATES_PER_GENERATOR_UPDATE):
        real_data_batch = Variable(next(uniform_generator))
        critic_noise_sample = Variable(utils.utils.noise(size=(real_data_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
        synthetic_data_batch = generator(critic_noise_sample)
        critic_loss = critic.train(real_data_batch, synthetic_data_batch, LAMBDA)
        running_critic_loss += critic_loss.data[0]

    # Train generator
    generator_noise_sample = Variable(utils.utils.noise(size=(real_data_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
    synthetic_data_batch = generator(generator_noise_sample)
    critic_output = critic(synthetic_data_batch)
    generator_loss = generator.train(critic_output)
    running_generator_loss += generator_loss.data[0]

    # Visualization
    if training_step % VISUALIZATION_INTERVAL == 0:
        # Timing
        running_batch_elapsed_time = timeit.default_timer() - running_batch_start_time
        running_batch_start_time = timeit.default_timer()

        num_training_batches_remaining = (TRAINING_STEPS - training_step) / BATCH_SIZE
        estimated_minutes_remaining = (num_training_batches_remaining * running_batch_elapsed_time) / 60.0

        print("===== TRAINING STEP {} | ~{:.0f} MINUTES REMAINING =====".format(training_step, estimated_minutes_remaining))
        print("CRITIC LOSS:     {0}".format(running_critic_loss))
        print("GENERATOR LOSS:  {0}\n".format(running_generator_loss))

        # Loss histories
        critic_losses_per_vis_interval.append(running_critic_loss)
        generator_losses_per_vis_interval.append(running_generator_loss)
        running_critic_loss = 0.0
        running_generator_loss = 0.0

        Plot.plot_histories([critic_losses_per_vis_interval],
                            ["Critic"],
                            "{0}critic_loss_history.png".format(MODEL_OUTPUT_DIR))
        Plot.plot_histories([generator_losses_per_vis_interval],
                            ["Generator"],
                            "{0}generator_loss_history.png".format(MODEL_OUTPUT_DIR))

        # Save model at checkpoint
        torch.save(generator.state_dict(), "{0}generator".format(MODEL_OUTPUT_DIR))
        torch.save(critic.state_dict(), "{0}critic".format(MODEL_OUTPUT_DIR))

        # Visualize samples
        Plot.plot_samples(real_data=real_data_batch.data.cpu().numpy(),
                          synthetic_data=synthetic_data_batch.data.cpu().numpy(),
                          output_file="{0}sample_{1}".format(VIS_OUTPUT_DIR, training_step))
