import argparse
import os
import random
import shutil
import sys

from preprocessor import Preprocessor

parser = argparse.ArgumentParser(description="Utility script that given a folder of Brainpedia data can generate two new folders representing a split of data with a tag and without a tag.")
parser.add_argument('data_dir', help='the directory containing Brainpedia data')
parser.add_argument('data_dir_cache', help='the directory to use as a cache for the Brainpedia data')
parser.add_argument('tag', help='the tag to split the data around')
parser.add_argument('output_dir_with_tag', help='the directory to output data with the tag')
parser.add_argument('output_dir_without_tag', help='the directory to output data without the tag')
args = parser.parse_args()

# Check for existence of data directory
if not os.path.isdir(args.data_dir):
    sys.exit("Could not find data directory at: '{0}'".format(args.data_dir))

# Collect filenames
data_dir_filenames = os.listdir(args.data_dir)
all_data_filenames = [fname for fname in data_dir_filenames if fname[-6:] == 'nii.gz']
labels_for_data_filename = [Preprocessor.labels_for_brain_image(args.data_dir+fname) for fname in all_data_filenames]

# Split filenames into files with tag and without
data_filenames_with_tag = []
data_filenames_without_tag = []
for data_filename, labels in zip(all_data_filenames, labels_for_data_filename):
    if args.tag in labels:
        data_filenames_with_tag.append(data_filename)
    else:
        data_filenames_without_tag.append(data_filename)

# Create output directories as needed:
if not os.path.isdir(args.output_dir_with_tag):
    os.makedirs(args.output_dir_with_tag)

if not os.path.isdir(args.output_dir_without_tag):
    os.makedirs(args.output_dir_without_tag)

# Copy data in filenames list + associated metadata from data dir to output dir
def copy_data(data_filenames, output_dir):
    for data_file_name in data_filenames:
        metadata_file_name = data_file_name.split('.')[0] + '_metadata.json'

        shutil.copyfile(args.data_dir + data_file_name, output_dir + data_file_name)
        shutil.copyfile(args.data_dir + metadata_file_name, output_dir + metadata_file_name)


copy_data(data_filenames_with_tag, args.output_dir_with_tag)
copy_data(data_filenames_without_tag, args.output_dir_without_tag)
