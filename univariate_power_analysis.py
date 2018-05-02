import argparse
import numpy as np
import os
import shutil

from scipy.stats import ttest_ind
from utils.mmd import mmd
from utils.sampling import *


# Parse arguments
parser = argparse.ArgumentParser(description="Compare classical two sample t test to non-parametric tests for real and synthetic univariate distributions.")
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
def power_calculations(d1, d2, n_1, n_2, alpha=0.05, k=50):
    # Use boostrap technique to estimate the distribution of the
    #  p-value statistic
    two_sample_t_test_p_value_dist = []
    mmd_rejections = []
    for br in range(k):
        d1_replicate = np.random.choice(d1, size=n_1, replace=True)
        d2_replicate = np.random.choice(d2, size=n_2, replace=True)

        # Classical two sample t test
        two_sample_t_test = ttest_ind(d1_replicate, d2_replicate, equal_var=True)
        two_sample_t_test_p_value_dist.append(two_sample_t_test.pvalue)

        # MMD statistic
        d1_replicate = np.expand_dims(d1_replicate, 1)
        d2_replicate = np.expand_dims(d2_replicate, 1)
        mmd_reject = mmd(d1_replicate, d2_replicate, sigma=None, alpha=alpha, k=100)
        mmd_rejections.append(mmd_reject)

    # Use monte carlo to estimate the power of a test with significance level of alpha
    #    => average number of p values less than alpha
    #    => average number of MMD rejections
    two_sample_t_test_power = np.mean([p < alpha for p in two_sample_t_test_p_value_dist])
    mmd_test_power = np.mean(mmd_rejections)
    return two_sample_t_test_power, mmd_test_power


# Compute power for various n
n = np.linspace(2, 100, num=25)

t_test_power_for_n = []
mmd_test_power_for_n = []
syn_t_test_power_for_n = []
syn_mmd_test_power_for_n = []
for i in range(len(n)):
    t_real, mmd_real = power_calculations(real_dataset_1, real_dataset_2, int(n[i]), int(n[i]))
    t_syn, mmd_syn = power_calculations(syn_dataset_1, syn_dataset_2, int(n[i]), int(n[i]))

    t_test_power_for_n.append(t_real)
    mmd_test_power_for_n.append(mmd_real)
    syn_t_test_power_for_n.append(t_syn)
    syn_mmd_test_power_for_n.append(mmd_syn)

# Save results to output dir
n = np.array(n)
t_test_real_power = np.array(t_test_power_for_n)
mmd_test_real_power = np.array(mmd_test_power_for_n)
t_test_syn_power = np.array(syn_t_test_power_for_n)
mmd_test_syn_power = np.array(syn_mmd_test_power_for_n)

np.save(args.output_dir+'n', n)
np.save(args.output_dir+'t_test_real_power', t_test_real_power)
np.save(args.output_dir+'mmd_test_real_power', mmd_test_real_power)
np.save(args.output_dir+'t_test_syn_power', t_test_syn_power)
np.save(args.output_dir+'mmd_test_syn_power', mmd_test_syn_power)
