import argparse
import numpy as np
import os
import random
import shutil

from utils.sampling import sample

# Parse arguments
parser = argparse.ArgumentParser(description="Draw specified number of samples from fMRI data directory and write to specified output directory.")
parser.add_argument('data_dir', help='the univariate probaility distribution from which data should be sampled')
parser.add_argument('num_samples', type=int, help='the number of samples to draw')
parser.add_argument('output_dir', help='the directory to save samples')
args = parser.parse_args()

# Check for existence of data directory
if not os.path.isdir(args.data_dir):
    sys.exit("Could not find data directory at: '{0}'".format(args.data_dir))

# Create output directories as needed
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# Collect all data filenames
data_dir_filenames = os.listdir(args.data_dir)
all_data_filenames = [fname for fname in data_dir_filenames if fname[-6:] == 'nii.gz']
random.shuffle(all_data_filenames)

# Select num_samples filenames
selected_data_filenames = all_data_filenames[:args.num_samples]

# Copy data in filenames list + associated metadata from data dir to output dir
def copy_data(data_filenames, output_dir):
    for data_file_name in data_filenames:
        metadata_file_name = data_file_name.split('.')[0] + '_metadata.json'

        shutil.copyfile(args.data_dir + data_file_name, output_dir + data_file_name)
        shutil.copyfile(args.data_dir + metadata_file_name, output_dir + metadata_file_name)


copy_data(selected_data_filenames, args.output_dir)
