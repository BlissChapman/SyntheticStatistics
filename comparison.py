import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from scipy.stats import ttest_ind
from mmd import mmd
from utils.sampling import *

# Parse arguments
parser = argparse.ArgumentParser(description="Compare classical two sample t test to non-parametric methods.")
parser.add_argument('dataset_1', help='the first synthetic datast to use in testing')
parser.add_argument('dataset_2', help='the second synthetic datast to use in testing')
parser.add_argument('output_dir', help='the directory to save training results')
args = parser.parse_args()

# Setup output directory
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# Load datasets
dataset_1 = np.random.normal(0, 1, 1000)
dataset_2 = np.random.normal(1, 1, 1000)

# dataset_1 = gaussian(1000)
# dataset_2 = gaussian(1000)

# dataset_2 = np.random.exponential(9, size=1000)

syn_dataset_1 = np.load(args.dataset_1)
syn_dataset_2 = np.load(args.dataset_2)

np.random.shuffle(dataset_1)
np.random.shuffle(dataset_2)
np.random.shuffle(syn_dataset_1)
np.random.shuffle(syn_dataset_2)

# Plot datasets
figure, axes = plt.subplots(nrows=3, ncols=1)
axes[0].hist(dataset_1, fc=(0, 0, 1, 0.5))
axes[0].hist(dataset_2, fc=(0.5, 0.5, 0.5, 0.5))
axes[0].set_title('Real Distributions')
axes[0].axes.yaxis.set_visible(False)

axes[1].hist(syn_dataset_1, fc=(0, 0, 1, 0.5))
axes[1].hist(syn_dataset_2, fc=(0.5, 0.5, 0.5, 0.5))
axes[1].set_title('Synthetic Distributions')
axes[1].axes.yaxis.set_visible(False)

# Power calculation
def power_calculations(d1, d2, n_1, n_2, alpha=0.05, k=10**3):
    # Use boostrap technique to estimate the distribution of the
    #  p-value statistic
    two_sample_t_test_p_value_dist = []
    mmd_test_stat_dist = []
    for br in range(k):
        dist_1_replicate = np.random.choice(d1, size=n_1, replace=True)
        dist_2_replicate = np.random.choice(d2, size=n_2, replace=True)

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

# Compute power for various n
n = np.linspace(2, 200, num=50)

t_test_power_for_n = []
mmd_test_power_for_n = []
syn_t_test_power_for_n = []
syn_mmd_test_power_for_n = []
for i in range(len(n)):
    t_real, mmd_real = power_calculations(dataset_1, dataset_2, int(n[i]), int(n[i]))
    t_syn, mmd_syn = power_calculations(syn_dataset_1, syn_dataset_2, int(n[i]), int(n[i]))

    t_test_power_for_n.append(t_real)
    mmd_test_power_for_n.append(mmd_real)
    syn_t_test_power_for_n.append(t_syn)
    syn_mmd_test_power_for_n.append(mmd_syn)

    print("PERCENT COMPLETE: {0:.2f}%\r".format(100*float(i)/float(len(n))), end='')


# Plot curve of n vs power
axes[2].plot(n, t_test_power_for_n, label='T Test Real')
axes[2].plot(n, syn_t_test_power_for_n, label='T Test Syn')
axes[2].plot(n, mmd_test_power_for_n, label='MMD Test Real')
axes[2].plot(n, syn_mmd_test_power_for_n, label='MMD Test Syn')
axes[2].set_title('Sample Size vs Power')
axes[2].set_xlabel('Sample Size')
axes[2].set_ylabel('Power')
axes[2].set_ylim([-0.1, 1.1])
axes[2].legend(loc="upper right")

# Save results
figure.tight_layout()
figure.savefig('{0}sample_size_vs_power.png'.format(args.output_dir))
