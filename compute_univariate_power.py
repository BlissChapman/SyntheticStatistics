import argparse
import numpy as np
import os
import shutil

from scipy.stats import ttest_ind
from utils.mmd import mmd
from utils.sampling import *


# Parse arguments
parser = argparse.ArgumentParser(description="Compute power of tests between real and synthetic univariate distributions.")
parser.add_argument('real_dataset_1', help='the path to the first real dataset')
parser.add_argument('syn_dataset_1', help='the path to the synthetic dataset generated from a model trained on real_dataset_1')
parser.add_argument('real_dataset_2', help='the path to the second real dataset')
parser.add_argument('syn_dataset_2', help='the path to the synthetic dataset generated from a model trained on real_dataset_2')
parser.add_argument('output_dir', help='the directory to save comparison results')
args = parser.parse_args()

# Setup output directory
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# Load datasets
real_dataset_1 = np.load(args.real_dataset_1)
real_dataset_2 = np.load(args.real_dataset_2)
syn_dataset_1 = np.load(args.syn_dataset_1)
syn_dataset_2 = np.load(args.syn_dataset_2)

np.random.shuffle(real_dataset_1)
np.random.shuffle(real_dataset_2)
np.random.shuffle(syn_dataset_1)
np.random.shuffle(syn_dataset_2)

# Power calculation
def power_calculations(d1, d2, n_1, n_2, alpha=0.05, k=10**3):
    # Use boostrap technique to estimate the distribution of the
    #  p-value statistic
    two_sample_t_test_p_value_dist = []
    #mmd_test_stat_dist = []
    for br in range(k):
        d1_replicate = np.random.choice(d1, size=n_1, replace=True)
        d2_replicate = np.random.choice(d2, size=n_2, replace=True)

        # Classical two sample t test
        two_sample_t_test = ttest_ind(d1_replicate, d2_replicate, equal_var=True)
        two_sample_t_test_p_value_dist.append(two_sample_t_test.pvalue)

        # MMD statistic
        # d1_replicate = np.expand_dims(d1_replicate, 1)
        # d2_replicate = np.expand_dims(d2_replicate, 1)
        # mmd_stat = mmd(d1_replicate, d2_replicate)[1]
        # mmd_test_stat_dist.append(mmd_stat)

    # Use monte carlo to estimate the power of a test with significance level 0.05
    #    => average number of p values less than alpha
    #    => average number of MMD statistics greater than alpha
    two_sample_t_test_power = np.mean([p < alpha for p in two_sample_t_test_p_value_dist])
    # mmd_test_power = np.mean([mmd_stat > alpha for mmd_stat in mmd_test_stat_dist])

    return two_sample_t_test_power  # , mmd_test_power


# Compute power for a test between the real and syn distributions respectively
t_real = power_calculations(real_dataset_1, real_dataset_2, real_dataset_1.shape[0], real_dataset_2.shape[0])
t_syn = power_calculations(syn_dataset_1, syn_dataset_2, syn_dataset_1.shape[0], syn_dataset_2.shape[0])

with open(args.output_dir + 'results.txt', 'w') as results_f:
    results_f.write("{0},{1}".format(t_real, t_syn))
