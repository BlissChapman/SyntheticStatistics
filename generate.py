import argparse
import numpy
import os
import shutil

from utils.sampling import *

# Parse arguments
parser = argparse.ArgumentParser(description="Generate specified number of samples from hardcoded distribution and write to specified output directory.")
parser.add_argument('num_samples', type=int, help='the number of samples to generate')
parser.add_argument('output_dir', help='the directory to save generated samples')
args = parser.parse_args()

# Housekeeping
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)
np.random.seed(1)

# Generate samples and save them to disk
data = gaussian(args.num_samples)
np.save(args.output_dir + 'data', data)
print("Data saved to '{0}'".format(args.output_dir + 'data'))
