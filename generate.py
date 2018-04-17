import argparse
import numpy as np
import os
import shutil

from utils.sampling import sample

# Parse arguments
parser = argparse.ArgumentParser(description="Generate specified number of samples from hardcoded distribution and write to specified output directory.")
parser.add_argument('distribution', choices=['gaussian_0', 'gaussian_0_1', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture'], help='the univariate probaility distribution from which data should be sampled')
parser.add_argument('num_samples', type=int, help='the number of samples to generate')
parser.add_argument('output_dir', help='the directory to save generated samples')
args = parser.parse_args()

# Housekeeping
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)
np.random.seed(1)

# Generate samples and save them to disk
data = sample(args.num_samples, args.distribution)
np.save(args.output_dir + 'data', data)
