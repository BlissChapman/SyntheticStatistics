import argparse
import numpy as np
import os
import shutil

from scipy.stats import ttest_ind
from utils.freqopttest.data import TSTData
from utils.freqopttest.kernel import KGauss
from utils.freqopttest.tst import LinearMMDTest
from utils.multiple_comparison import conservative_mmd, fdr_correction
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

# Initialize MMD test
mmd_kernel = KGauss(sigma2=1.0)
mmd = LinearMMDTest(kernel=mmd_kernel, alpha=0.05)

def fdr_t_test_power(d1, d2, n_1, n_2, alpha=0.05, k=10**1):
    fdr_rejections = []

    for br in range(k):
        d1_idx = np.random.randint(low=0, high=d1.shape[0], size=n_1)
        d2_idx = np.random.randint(low=0, high=d2.shape[0], size=n_2)

        d1_replicate = d1[d1_idx].squeeze()
        d2_replicate = d2[d2_idx].squeeze()

        # FDR corrected univariate tests
        two_sample_t_test_p_vals_by_dim = np.zeros(d1.shape[1:])
        for i in range(two_sample_t_test_p_vals_by_dim.shape[0]):
            d1_vals = d1[:, i]
            d2_vals = d2[:, i]
            two_sample_t_test_p_vals_by_dim[i] = ttest_ind(d1_vals, d2_vals, equal_var=True).pvalue

        fdr_reject_by_dim = fdr_correction(two_sample_t_test_p_vals_by_dim, alpha=alpha)[0]
        fdr_reject = sum(fdr_reject_by_dim) > 0  # reject if any dim rejects
        fdr_rejections.append(fdr_reject)

    fdr_power = np.mean(fdr_rejections)
    return fdr_power

def avg_p_val(p, ps, k=1000):
    pvals = np.zeros(k)
    for k_i in range(k):
        indices = np.arange(p.shape[0])
        replicate_1_indices = np.random.choice(indices, size=p.shape[0], replace=True)
        replicate_2_indices = np.random.choice(indices, size=p.shape[0], replace=True)

        d1 = p[replicate_1_indices]
        d2 = ps[replicate_2_indices]
        pval = mmd.perform_test(TSTData(d1, d2))['pvalue']
        pvals[k_i] = pval
    return np.mean(pvals)

def power_calculations(d1, d2, conservative_adjustment, n_1, n_2, alpha=0.05, k=50):
    # FDR corrected T-test power
    fdr_power = fdr_t_test_power(d1, d2, n_1, n_2, alpha=alpha, k=k)

    # Use boostrap technique to estimate the power of the MMD test
    mmd_rejections = []
    conservative_mmd_rejections = []
    for br in range(k):
        d1_indices = np.arange(n_1)
        d2_indices = np.arange(n_2)

        d1_replicate_indices = np.random.choice(d1_indices, size=n_1, replace=True)
        d2_replicate_indices = np.random.choice(d2_indices, size=n_2, replace=True)

        d1_replicate = d1[d1_replicate_indices]
        d2_replicate = d2[d2_replicate_indices]

        # MMD statistic
        tst_data = TSTData(d1_replicate, d2_replicate)
        mmd_reject = mmd.perform_test(tst_data)['h0_rejected']
        mmd_rejections.append(mmd_reject)

        # Conservative MMD statistic:
        conservative_mmd_reject = conservative_mmd(d1_replicate, d2_replicate, conservative_adjustment, alpha)
        conservative_mmd_rejections.append(conservative_mmd_reject)

    mmd_test_power = np.mean(mmd_rejections)
    conservative_mmd_test_power = np.mean(conservative_mmd_rejections)
    return fdr_power, mmd_test_power, conservative_mmd_test_power

# Estimate conservative adjustment
real_dataset_1_null_pval = avg_p_val(real_dataset_1, real_dataset_1)
real_dataset_2_null_pval = avg_p_val(real_dataset_2, real_dataset_2)

REAL = np.concatenate((real_dataset_1, real_dataset_2))
SYN = np.concatenate((syn_dataset_1, syn_dataset_2))
REAL_SYN_AVG_PVAL = avg_p_val(REAL, SYN)

conservative_adjustment = real_dataset_1_null_pval + real_dataset_2_null_pval - 2*REAL_SYN_AVG_PVAL
conservative_adjustment = 0.15*conservative_adjustment

# Compute power for various n
n = np.linspace(10, 1000, num=50)

fdr_test_power_for_n = []
mmd_test_power_for_n = []
syn_fdr_test_power_for_n = []
syn_mmd_test_power_for_n = []
syn_mmd_conservative_test_power_for_n = []
for i in range(len(n)):
    fdr_real, mmd_real, _ = power_calculations(real_dataset_1, real_dataset_2, 0.0, int(n[i]), int(n[i]))
    fdr_syn, mmd_syn, conservative_mmd_syn = power_calculations(syn_dataset_1, syn_dataset_2, conservative_adjustment, int(n[i]), int(n[i]))

    fdr_test_power_for_n.append(fdr_real)
    mmd_test_power_for_n.append(mmd_real)
    syn_fdr_test_power_for_n.append(fdr_syn)
    syn_mmd_test_power_for_n.append(mmd_syn)
    syn_mmd_conservative_test_power_for_n.append(conservative_mmd_syn)

# Save results to output dir
n = np.array(n)
fdr_test_real_power = np.array(fdr_test_power_for_n)
mmd_test_real_power = np.array(mmd_test_power_for_n)
fdr_test_syn_power = np.array(syn_fdr_test_power_for_n)
mmd_test_syn_power = np.array(syn_mmd_test_power_for_n)
mmd_conservative_test_syn_power = np.array(syn_mmd_conservative_test_power_for_n)

np.save(args.output_dir+'n', n)
np.save(args.output_dir+'fdr_test_real_power', fdr_test_real_power)
np.save(args.output_dir+'mmd_test_real_power', mmd_test_real_power)
np.save(args.output_dir+'fdr_test_syn_power', fdr_test_syn_power)
np.save(args.output_dir+'mmd_test_syn_power', mmd_test_syn_power)
np.save(args.output_dir+'mmd_conservative_test_syn_power', mmd_conservative_test_syn_power)
