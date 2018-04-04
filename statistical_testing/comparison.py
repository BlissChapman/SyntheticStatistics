import argparse
import numpy as np
import os
import shutil

from scipy.stats import ttest_ind
from mmd import mmd

# Parse arguments
parser = argparse.ArgumentParser(description="Compare classical two sample t test to non-parametric methods.")
parser.add_argument('dataset_1', help='the first datast to use in testing')
parser.add_argument('dataset_2', help='the second datast to use in testing')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# Setup output directory
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# Load datasets
dataset_1 = np.random.chisquare(9, size=1000)#np.load(args.dataset_1)[0:1000]
dataset_2 = np.random.exponential(scale=9.0, size=1000)#np.load(args.dataset_2)[0:1000]

# Power calculation
def power_calculations(dataset_1, dataset_2, alpha=0.05, k=10):
    n_1 = len(dataset_1)
    n_2 = len(dataset_2)

    # Use boostrap technique to estimate the distribution of the
    #  p-value statistic
    two_sample_t_test_p_value_dist = []
    mmd_test_stat_dist = []
    for br in range(k):
        dist_1_replicate = np.random.choice(dataset_1, size=n_1, replace=True)
        dist_2_replicate = np.random.choice(dataset_2, size=n_2, replace=True)

        # Classical two sample t test
        two_sample_t_test = ttest_ind(dist_1_replicate, dist_2_replicate, equal_var=True)
        two_sample_t_test_p_value_dist.append(two_sample_t_test.pvalue)

        # MMD statistic
        dist_1_replicate = np.expand_dims(dist_1_replicate, 1)
        dist_2_replicate = np.expand_dims(dist_2_replicate, 1)
        mmd_stat = mmd(dist_1_replicate, dist_2_replicate)[1]
        mmd_test_stat_dist.append(mmd_stat)

    # Use monte carlo to estimate the power of a test with significance level 0.05
    #    => average number of p values less than alpha
    #    => average number of MMD statistics greater than alpha
    two_sample_t_test_power = np.mean([p < alpha for p in two_sample_t_test_p_value_dist])
    mmd_test_power = np.mean([mmd_stat > alpha for mmd_stat in mmd_test_stat_dist])

    return two_sample_t_test_power, mmd_test_power

# Perform power calculations
t_test_power, mmd_test_power = power_calculations(dataset_1, dataset_2)
print("========== TEST POWER ==========")
print("TWO SAMPLE T TEST:     {0}".format(t_test_power))
print("MMD TEST:              {0}".format(mmd_test_power))
