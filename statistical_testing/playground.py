import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy.stats import ttest_ind
from mmd import mmd

alpha = 0.05
n_1 = 10**3
n_2 = 10**3
num_bootstrap_replicates = 5
alt_thetas = np.linspace(-2.0, 2.0, 15)
figure = plt.figure(figsize=(10, 30))

# Simulate alternate means and compute power of a test between distributions
two_sample_t_test_power_for_alt_theta = []
mmd_test_power_for_alt_theta = []

for i in range(len(alt_thetas)):
    # Sample from dist_1 and dist_2
    dist_1 = np.random.normal(0, 1, n_1)
    dist_2 = np.random.normal(alt_thetas[i], 1, n_2)

    # dist_1 = np.random.chisquare(9, size=n_1)
    # dist_2 = np.random.exponential(scale=alt_thetas[i], size=n_2)

    # Plot samples
    dist_ax = plt.subplot(len(alt_thetas)+1, 1, i+1)
    dist_ax.hist(dist_1, fc=(0, 0, 1, 0.5))
    dist_ax.hist(dist_2, fc=(0.5, 0.5, 0.5, 0.5))

    # Use boostrap technique to estimate the distribution of the
    #  p-value statistic
    two_sample_t_test_p_value_dist = []
    mmd_test_stat_dist = []
    for br in range(num_bootstrap_replicates):
        dist_1_replicate = np.random.choice(dist_1, size=n_1, replace=True)
        dist_2_replicate = np.random.choice(dist_2, size=n_2, replace=True)

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
    two_sample_t_test_power_for_alt_theta.append(two_sample_t_test_power)

    mmd_test_power = np.mean([mmd_stat > alpha for mmd_stat in mmd_test_stat_dist])
    mmd_test_power_for_alt_theta.append(mmd_test_power)

    # Print % complete
    print("PERCENT COMPLETE: {0:.2f}%\r".format((i+1)*100/len(alt_thetas)), end='')

# Plot power of classical two sample t test as a function of alternative mean
power_ax = plt.subplot(len(alt_thetas)+1, 1, len(alt_thetas)+1)
power_ax.plot(alt_thetas, two_sample_t_test_power_for_alt_theta)
power_ax.plot(alt_thetas, mmd_test_power_for_alt_theta)
power_ax.set_xlabel('Alternative Means')
power_ax.set_ylabel('Power of Tests')
power_ax.set_ylim([-0.1, 1.1])
power_ax.legend("TM",loc="upper right")
figure.savefig('OUTPUT/test.png')
