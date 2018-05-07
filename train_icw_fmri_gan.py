import matplotlib
matplotlib.use('Agg')

import argparse
import datetime
import models.ICW_FMRI_GAN
import nibabel
import numpy as np
import os
import shutil
import timeit
import torch

from brainpedia.brainpedia import Brainpedia
from brainpedia.fmri_processing import invert_preprocessor_scaling
from torch.autograd import Variable
from utils.sampling import noise
from utils.plot import Plot


parser = argparse.ArgumentParser(description="Train ICW_FMRI_GAN.")
parser.add_argument('train_data_dir', help='the directory containing real fMRI data to train on')
parser.add_argument('train_data_dir_cache', help='the directory to use as a cache for the train_data_dir preprocessing')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# ========== OUTPUT DIRECTORIES ==========
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# ========== Hyperparameters ==========
DOWNSAMPLE_SCALE = 0.25
MULTI_TAG_LABEL_ENCODING = True
TRAINING_STEPS = 200000
BATCH_SIZE = 50
MODEL_DIMENSIONALITY = 64
CONDITONING_DIMENSIONALITY = 5
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 1
LAMBDA = 10
NOISE_SAMPLE_LENGTH = 128

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()

np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== Data ==========
brainpedia = Brainpedia(data_dirs=[args.train_data_dir],
                        cache_dir=args.train_data_dir_cache,
                        scale=DOWNSAMPLE_SCALE,
                        multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
all_brain_data, all_brain_data_tags = brainpedia.all_data()

brainpedia_generator = Brainpedia.batch_generator(all_brain_data, all_brain_data_tags, BATCH_SIZE, CUDA)
brain_data_shape, brain_data_tag_shape = brainpedia.sample_shapes()

# ========== Models ==========
generator = models.ICW_FMRI_GAN.Generator(input_size=NOISE_SAMPLE_LENGTH,
                                          output_shape=brain_data_shape,
                                          dimensionality=MODEL_DIMENSIONALITY,
                                          num_classes=brain_data_tag_shape[0],
                                          conditioning_dimensionality=CONDITONING_DIMENSIONALITY,
                                          cudaEnabled=CUDA)
critic = models.ICW_FMRI_GAN.Critic(dimensionality=MODEL_DIMENSIONALITY,
                                    num_classes=brain_data_tag_shape[0],
                                    conditioning_dimensionality=CONDITONING_DIMENSIONALITY,
                                    cudaEnabled=CUDA)

# ========= Training =========
for training_step in range(1, TRAINING_STEPS + 1):
    # Train critic
    for critic_step in range(CRITIC_UPDATES_PER_GENERATOR_UPDATE):
        real_brain_img_data_batch, labels_batch = next(brainpedia_generator)
        real_brain_img_data_batch = Variable(real_brain_img_data_batch)
        labels_batch = Variable(labels_batch)

        noise_sample_c = Variable(noise(size=(labels_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
        synthetic_brain_img_data_batch = generator(noise_sample_c, labels_batch)
        _ = critic.train(real_brain_img_data_batch, synthetic_brain_img_data_batch, labels_batch, LAMBDA)

    # Train generator
    noise_sample_g = Variable(noise(size=(labels_batch.shape[0], NOISE_SAMPLE_LENGTH), cuda=CUDA))
    synthetic_brain_img_data_batch = generator(noise_sample_g, labels_batch)
    critic_output = critic(synthetic_brain_img_data_batch, labels_batch)
    _ = generator.train(critic_output)

    if training_step % 10000 == 0:
        # Save model at checkpoint
        torch.save(generator.state_dict(), "{0}generator".format(args.output_dir))
        torch.save(critic.state_dict(), "{0}critic".format(args.output_dir))
        
# Save model at checkpoint
torch.save(generator.state_dict(), "{0}generator".format(args.output_dir))
torch.save(critic.state_dict(), "{0}critic".format(args.output_dir))
