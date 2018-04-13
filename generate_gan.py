import argparse
import datetime
import models.ProbabilityDistGAN
import numpy as np
import os
import shutil
import torch

from torch.autograd import Variable
from utils.sampling import uniform_noise

parser = argparse.ArgumentParser(description="Generate specified number of samples from trained generator and write to specified output directory.")
parser.add_argument('generator_state_dict_path', help='path to a file containing the generative model state dict')
parser.add_argument('num_samples', type=int, help='the number of samples to generate')
parser.add_argument('output_dir', help='the directory to save generated samples')
args = parser.parse_args()

# ========== HOUSEKEEPING ==========
CUDA = torch.cuda.is_available()
np.random.seed(1)
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# ========== Hyperparameters ==========
BATCH_SIZE = 16
MODEL_DIMENSIONALITY = 64
SAMPLE_LENGTH = 1
NOISE_SAMPLE_LENGTH = 64

description_f = open(args.output_dir + 'description.txt', 'w')
description_f.write('DATE: {0}\n\n'.format(datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')))
description_f.write('BATCH_SIZE: {0}\n'.format(BATCH_SIZE))
description_f.write('MODEL_DIMENSIONALITY: {0}\n'.format(MODEL_DIMENSIONALITY))
description_f.write('SAMPLE_LENGTH: {0}\n'.format(SAMPLE_LENGTH))
description_f.write('NOISE_SAMPLE_LENGTH: {0}\n'.format(NOISE_SAMPLE_LENGTH))
description_f.close()

# ========== Models ==========
generator = models.ProbabilityDistGAN.Generator(input_width=NOISE_SAMPLE_LENGTH,
                                                output_width=SAMPLE_LENGTH,
                                                dimensionality=MODEL_DIMENSIONALITY,
                                                cudaEnabled=CUDA)
generator.load_state_dict(torch.load(args.generator_state_dict_path))

# ========== Sample Generation ==========
synthetic_data = []
for step in range(args.num_samples):
    # Generate synthetic data
    noise_sample = Variable(uniform_noise(NOISE_SAMPLE_LENGTH, BATCH_SIZE, CUDA))
    synthetic_data_sample = generator(noise_sample).data.numpy().flatten()
    synthetic_data_sample = list(synthetic_data_sample)
    synthetic_data += synthetic_data_sample

# Write samples to disk
synthetic_data = np.array(synthetic_data)
np.save(args.output_dir + 'data', synthetic_data)
