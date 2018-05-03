import argparse
import numpy as np
import os
import shutil

from utils.freqopttest.tst import LinearMMDTest
from utils.mmd import mmd
from utils.multiple_comparison import multivariate_power_calculation
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
    # Use boostrap technique to estimate the power of the MMD and
    #  fdr corrected T-test
    fdr_power_for_k = []
    mmd_rejections = []
    for br in range(k):
        d1_indices = np.arange(n_1)
        d2_indices = np.arange(n_2)
        d1_replicate_indices = np.random.choice(d1_indices, size=n_1, replace=True)
        d2_replicate_indices = np.random.choice(d2_indices, size=n_2, replace=True)

        d1_replicate = d1[d1_replicate_indices]
        d2_replicate = d2[d2_replicate_indices]

        # FDR corrected t-tests

        fdr_power = multivariate_power_calculation(d1_replicate, d2_replicate, n_1, n_2, alpha=alpha, k=100)
        fdr_power_for_k.append(fdr_power)

        # MMD statistic
        mmd_reject = mmd(d1_replicate, d2_replicate, sigma=None, alpha=alpha, k=100)
        mmd_rejections.append(mmd_reject)

    fdr_test_power = np.mean(fdr_power_for_k)
    mmd_test_power = np.mean(mmd_rejections)
    return fdr_test_power, mmd_test_power


# Compute power for various n
n = np.linspace(10, 100, num=18)

fdr_test_power_for_n = []
mmd_test_power_for_n = []
syn_fdr_test_power_for_n = []
syn_mmd_test_power_for_n = []
for i in range(len(n)):
    fdr_real, mmd_real = power_calculations(real_dataset_1, real_dataset_2, int(n[i]), int(n[i]))
    fdr_syn, mmd_syn = power_calculations(syn_dataset_1, syn_dataset_2, int(n[i]), int(n[i]))

    fdr_test_power_for_n.append(fdr_real)
    mmd_test_power_for_n.append(mmd_real)
    syn_fdr_test_power_for_n.append(fdr_syn)
    syn_mmd_test_power_for_n.append(mmd_syn)

# Save results to output dir
n = np.array(n)
fdr_test_real_power = np.array(fdr_test_power_for_n)
mmd_test_real_power = np.array(mmd_test_power_for_n)
fdr_test_syn_power = np.array(syn_fdr_test_power_for_n)
mmd_test_syn_power = np.array(syn_mmd_test_power_for_n)

np.save(args.output_dir+'n', n)
np.save(args.output_dir+'fdr_test_real_power', fdr_test_real_power)
np.save(args.output_dir+'mmd_test_real_power', mmd_test_real_power)
np.save(args.output_dir+'fdr_test_syn_power', fdr_test_syn_power)
np.save(args.output_dir+'mmd_test_syn_power', mmd_test_syn_power)
