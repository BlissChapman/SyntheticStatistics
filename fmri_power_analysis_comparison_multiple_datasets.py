import matplotlib
matplotlib.use('Agg')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil

from brainpedia.brainpedia import Brainpedia
from brainpedia.fmri_processing import invert_preprocessor_scaling
from utils.multiple_comparison import bootstrap_rejecting_voxels_mask, fmri_power_calculations
from nilearn import plotting
from utils.sampling import *


# Parse arguments
parser = argparse.ArgumentParser(description="Compare classical two sample t test to non-parametric tests for real and synthetic fMRI brain imaging datasets.")
parser.add_argument('real_dataset_1_dir', help='the directory containing the first real fMRI dataset')
parser.add_argument('real_dataset_1_cache_dir', help='the directory to use as a cache for real dataset 1 preprocessing')
parser.add_argument('syn_dataset_1_dir', help='the directory containing the synthetic fMRI dataset generated from a model trained on real dataset 1')
parser.add_argument('syn_dataset_1_cache_dir', help='the directory to use as a cache for synthetic dataset 1 preprocessing')
parser.add_argument('dataset_1_label', help='the label to use when describing contents of dataset 1')
parser.add_argument('real_dataset_2_dir', help='the directory containing the second real fMRI dataset')
parser.add_argument('real_dataset_2_cache_dir', help='the directory to use as a cache for real dataset 2 preprocessing')
parser.add_argument('syn_dataset_2_dir', help='the directory containing the synthetic fMRI dataset generated from a model trained on real dataset 2')
parser.add_argument('syn_dataset_2_cache_dir', help='the directory to use as a cache for synthetic dataset 2 preprocessing')
parser.add_argument('dataset_2_label', help='the label to use when describing contents of dataset 1')
parser.add_argument('output_dir', help='the directory to save power analysis results')
args = parser.parse_args()

OUTPUT_DATA_DIR = args.output_dir + 'data/'
# Setup output directory
# shutil.rmtree(args.output_dir, ignore_errors=True)
try:
    os.makedirs(args.output_dir)
except:
    pass

try:
    os.makedirs(OUTPUT_DATA_DIR)
except:
    pass

# Load datasets
DOWNSAMPLE_SCALE = 0.25
MULTI_TAG_LABEL_ENCODING = False

real_dataset_1_brainpedia = Brainpedia(data_dirs=[args.real_dataset_1_dir],
                                       cache_dir=args.real_dataset_1_cache_dir,
                                       scale=DOWNSAMPLE_SCALE,
                                       multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
syn_dataset_1_brainpedia = Brainpedia(data_dirs=[args.syn_dataset_1_dir],
                                      cache_dir=args.syn_dataset_1_cache_dir,
                                      scale=DOWNSAMPLE_SCALE,
                                      multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
real_dataset_2_brainpedia = Brainpedia(data_dirs=[args.real_dataset_2_dir],
                                       cache_dir=args.real_dataset_2_cache_dir,
                                       scale=DOWNSAMPLE_SCALE,
                                       multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
syn_dataset_2_brainpedia = Brainpedia(data_dirs=[args.syn_dataset_2_dir],
                                      cache_dir=args.syn_dataset_2_cache_dir,
                                      scale=DOWNSAMPLE_SCALE,
                                      multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)

real_dataset_1, _ = real_dataset_1_brainpedia.all_data()
syn_dataset_1, _ = syn_dataset_1_brainpedia.all_data()
real_dataset_2, _ = real_dataset_2_brainpedia.all_data()
syn_dataset_2, _ = syn_dataset_2_brainpedia.all_data()

# Trim real datasets to the same length
real_dataset_length = min(len(real_dataset_1), len(real_dataset_2))
real_dataset_1 = np.array(real_dataset_1[:real_dataset_length])
real_dataset_2 = np.array(real_dataset_2[:real_dataset_length])

# Trim synthetic datasets to the same length
syn_dataset_length = min(len(syn_dataset_1), len(syn_dataset_2))
syn_dataset_1 = np.array(syn_dataset_1[:syn_dataset_length])
syn_dataset_2 = np.array(syn_dataset_2[:syn_dataset_length])

# Plot examples from datasets
real_dataset_1_img = invert_preprocessor_scaling(
    real_dataset_1[0].squeeze(), real_dataset_1_brainpedia.preprocessor)
real_dataset_2_img = invert_preprocessor_scaling(
    real_dataset_2[0].squeeze(), real_dataset_2_brainpedia.preprocessor)
syn_dataset_1_img = invert_preprocessor_scaling(
    syn_dataset_1[0].squeeze(), syn_dataset_2_brainpedia.preprocessor)
syn_dataset_2_img = invert_preprocessor_scaling(
    syn_dataset_2[2].squeeze(), syn_dataset_2_brainpedia.preprocessor)

figure, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 40))
plotting.plot_glass_brain(real_dataset_1_img, threshold='auto',
                          title="[REAL {0}]".format(args.dataset_1_label), axes=axes[0])
plotting.plot_glass_brain(syn_dataset_1_img, threshold='auto',
                          title="[SYN {0}]".format(args.dataset_1_label), axes=axes[1])
plotting.plot_glass_brain(real_dataset_2_img, threshold='auto',
                          title="[REAL {0}]".format(args.dataset_2_label), axes=axes[2])
plotting.plot_glass_brain(syn_dataset_2_img, threshold='auto',
                          title="[SYN {0}]".format(args.dataset_2_label), axes=axes[3])

# Compute statistical significance weights of each voxel in non-visual vs
# visual
num_trials = 5
k = 10
real_rejecting_voxels_mask = bootstrap_rejecting_voxels_mask(
    real_dataset_1.squeeze(), real_dataset_2.squeeze(), k=k)

# Compute power for various n
n = np.linspace(10, 100, num=25)
fdr_test_p_values_for_n = np.zeros((len(n), num_trials, k))
syn_fdr_test_p_values_for_n = np.zeros((len(n), num_trials, k))
mmd_test_p_values_for_n = np.zeros((len(n), num_trials, k))
syn_mmd_test_p_values_for_n = np.zeros((len(n), num_trials, k))

fdr_test_power_for_n = np.zeros((len(n), num_trials))
syn_fdr_test_power_for_n = np.zeros((len(n), num_trials))
mmd_test_power_for_n = np.zeros((len(n), num_trials))
syn_mmd_test_power_for_n = np.zeros((len(n), num_trials))

percent_rejecting_voxels_syn_for_n = np.zeros((len(n), k))
percent_rejecting_voxels_real_for_n = np.zeros((len(n), k))

wtp_syn_for_n = np.zeros((len(n), k))
wtn_syn_for_n = np.zeros((len(n), k))
wfp_syn_for_n = np.zeros((len(n), k))
wfn_syn_for_n = np.zeros((len(n), k))

wtp_real_for_n = np.zeros((len(n), k))
wtn_real_for_n = np.zeros((len(n), k))
wfp_real_for_n = np.zeros((len(n), k))
wfn_real_for_n = np.zeros((len(n), k))

for i in range(len(n)):
    # Determine sample sizes to draw from synthetic and real datasets
    # Note: there is limited real data. When there is none left, simply use the
    #   max available amount of data.
    syn_n = int(n[i])
    real_n = min(real_dataset_1.shape[0], int(n[i]))

    for t in range(num_trials):
        fdr_real_p_val, mmd_p_val, fdr_real_power, mmd_power, percent_rejecting_voxels_real, wtp_real, wtn_real, wfp_real, wfn_real = fmri_power_calculations(
            real_dataset_1, real_dataset_2, real_n, real_n, real_rejecting_voxels_mask, k=k)
        fdr_syn_p_val, mmd_syn_p_val, fdr_syn_power, mmd_syn_power, percent_rejecting_voxels_syn, wtp_syn, wtn_syn, wfp_syn, wfn_syn = fmri_power_calculations(
            syn_dataset_1, syn_dataset_2, syn_n, syn_n, real_rejecting_voxels_mask, k=k)

        fdr_test_p_values_for_n[i][t][:] = fdr_real_p_val[:]
        syn_fdr_test_p_values_for_n[i][t][:] = fdr_syn_p_val[:]
        mmd_test_p_values_for_n[i][t][:] = mmd_p_val[:]
        syn_mmd_test_p_values_for_n[i][t][:] = mmd_syn_p_val[:]

        fdr_test_power_for_n[i][t] = fdr_real_power
        syn_fdr_test_power_for_n[i][t] = fdr_syn_power
        mmd_test_power_for_n[i][t] = mmd_power
        syn_mmd_test_power_for_n[i][t] = mmd_syn_power

    percent_rejecting_voxels_syn_for_n[i][:] = percent_rejecting_voxels_syn
    percent_rejecting_voxels_real_for_n[i][:] = percent_rejecting_voxels_real

    wtp_syn_for_n[i][:] = wtp_syn[:]
    wtn_syn_for_n[i][:] = wtn_syn[:]
    wfp_syn_for_n[i][:] = wfp_syn[:]
    wfn_syn_for_n[i][:] = wfn_syn[:]

    wtp_real_for_n[i][:] = wtp_real[:]
    wtn_real_for_n[i][:] = wtn_real[:]
    wfp_real_for_n[i][:] = wfp_real[:]
    wfn_real_for_n[i][:] = wfn_real[:]

    print("PERCENT COMPLETE: {0:.2f}%\r".format(
        100 * float(i + 1) / float(len(n))), end='')

# Calculate Beta value for every trial and every sample size
def compute_beta(real_pvals, syn_pvals, alpha=0.05, k=50):
    l = 0.0
    h = 1.0

    for _ in range(k):
        beta = (l + h) / 2.0

        syn_reject_too_often = False
        for n in range(real_pvals.shape[0]):
            for trial in range(real_pvals.shape[1]):
                avg_real_rejection = np.mean(real_pvals[n][trial] < alpha)
                avg_syn_rejection = np.mean(syn_pvals[n][trial] + beta < alpha)
                if avg_syn_rejection > avg_real_rejection:
                    syn_reject_too_often = True

        if syn_reject_too_often:
            l = beta
        else:
            h = beta

    return beta

fdr_beta = 0.0443541875
mmd_beta = 0.0277111875

computed_fdr_beta = compute_beta(fdr_test_p_values_for_n, syn_fdr_test_p_values_for_n)
computed_mmd_beta = compute_beta(mmd_test_p_values_for_n, syn_mmd_test_p_values_for_n)

#((len(n), num_trials, k))
conservative_syn_fdr_test_p_values_for_n = np.copy(
    syn_fdr_test_p_values_for_n) + fdr_beta
conservative_syn_fdr_test_power = np.mean(
    conservative_syn_fdr_test_p_values_for_n < 0.05, axis=2)

conservative_syn_mmd_test_p_values_for_n = np.copy(
    syn_mmd_test_p_values_for_n) + mmd_beta
conservative_syn_mmd_test_power = np.mean(
    conservative_syn_mmd_test_p_values_for_n < 0.05, axis=2)

# Plot curve of n vs FDR corrected t test power
sns.tsplot(data=fdr_test_power_for_n.T, time=n, ci=[
           68, 95], color='blue', condition='REAL', ax=axes[4])
sns.tsplot(data=syn_fdr_test_power_for_n.T, time=n, ci=[
           68, 95], color='orange', condition='SYN', ax=axes[4])
sns.tsplot(data=conservative_syn_fdr_test_power.T, time=n, ci=[
           68, 95], color='green', condition='SYN CONSERVATIVE', ax=axes[4])
axes[4].set_title('Sample Size vs FDR Corrected T Test Power')
axes[4].set_xlabel('Sample Size, Applied Beta = %f, Computed Beta = %f' % (fdr_beta, computed_fdr_beta))
axes[4].set_ylabel('Power')
axes[4].set_ylim([-0.1, 1.1])
axes[4].legend(loc="upper right")
np.save('{0}[fmri_power_analysis]_[{1}_VS_{2}]_fdr_power_real.npy'.format(
    OUTPUT_DATA_DIR, args.dataset_1_label, args.dataset_2_label), fdr_test_power_for_n)
np.save('{0}[fmri_power_analysis]_[{1}_VS_{2}]_fdr_power_syn.npy'.format(
    OUTPUT_DATA_DIR, args.dataset_1_label, args.dataset_2_label), syn_fdr_test_power_for_n)
np.save('{0}[fmri_power_analysis]_[{1}_VS_{2}]_fdr_power_syncon.npy'.format(
    OUTPUT_DATA_DIR, args.dataset_1_label, args.dataset_2_label), conservative_syn_fdr_test_power)

# Plot curve of n vs MMD test power
sns.tsplot(data=mmd_test_power_for_n.T, time=n, ci=[
           68, 95], color='blue', condition='REAL', ax=axes[5])
sns.tsplot(data=syn_mmd_test_power_for_n.T, time=n, ci=[
           68, 95], color='orange', condition='SYN', ax=axes[5])
sns.tsplot(data=conservative_syn_mmd_test_power.T, time=n, ci=[
           68, 95], color='green', condition='SYN CONSERVATIVE', ax=axes[5])
axes[5].set_title('Sample Size vs MMD Test Power')
axes[5].set_xlabel('Sample Size, Applied Beta = %f, Computed Beta = %f' % (mmd_beta, computed_mmd_beta))
axes[5].set_ylabel('Power')
axes[5].set_ylim([-0.1, 1.1])
axes[5].legend(loc="upper right")

np.save('{0}[fmri_power_analysis]_[{1}_VS_{2}]_mmd_power_real.npy'.format(
    OUTPUT_DATA_DIR, args.dataset_1_label, args.dataset_2_label), mmd_test_power_for_n)
np.save('{0}[fmri_power_analysis]_[{1}_VS_{2}]_mmd_power_syn.npy'.format(
    OUTPUT_DATA_DIR, args.dataset_1_label, args.dataset_2_label), syn_mmd_test_power_for_n)
np.save('{0}[fmri_power_analysis]_[{1}_VS_{2}]_mmd_power_syncon.npy'.format(
    OUTPUT_DATA_DIR, args.dataset_1_label, args.dataset_2_label), conservative_syn_mmd_test_power)
# # Plot curve of percent rejecting voxels
# sns.tsplot(data=percent_rejecting_voxels_real_for_n.T, time=n, ci=[
#            68, 95], color='blue', condition='REAL', ax=axes[6])
# sns.tsplot(data=percent_rejecting_voxels_syn_for_n.T, time=n, ci=[
#            68, 95], color='orange', condition='SYN', ax=axes[6])
# axes[6].set_title('Sample Size vs Percent Significant Voxels')
# axes[6].set_xlabel('Sample Size')
# axes[6].set_ylabel('% Sig Voxels')
# axes[6].legend(loc="upper right")

# # Plot curve of n vs rejection overlaps
# # True Positive
# sns.tsplot(data=wtp_real_for_n.T, time=n, ci=[
#            68, 95], color='blue', condition='REAL', ax=axes[7])
# sns.tsplot(data=wtp_syn_for_n.T, time=n, ci=[
#            68, 95], color='orange', condition='SYN', ax=axes[7])
# axes[7].set_title('Sample Size vs Weighted True Positive')
# axes[7].set_xlabel('Sample Size')
# axes[7].set_ylabel('W_TP')
# axes[7].legend(loc="upper right")

# # True Negative
# sns.tsplot(data=wtn_real_for_n.T, time=n, ci=[
#            68, 95], color='blue', condition='REAL', ax=axes[8])
# sns.tsplot(data=wtn_syn_for_n.T, time=n, ci=[
#            68, 95], color='orange', condition='SYN', ax=axes[8])
# axes[8].set_title('Sample Size vs Weighted True Negatives')
# axes[8].set_xlabel('Sample Size')
# axes[8].set_ylabel('W_TN')
# axes[8].legend(loc="upper right")

# # False Positive
# sns.tsplot(data=wfp_real_for_n.T, time=n, ci=[
#            68, 95], color='blue', condition='REAL', ax=axes[9])
# sns.tsplot(data=wfp_syn_for_n.T, time=n, ci=[
#            68, 95], color='orange', condition='SYN', ax=axes[9])
# axes[9].set_title('Sample Size vs Weighted False Positives')
# axes[9].set_xlabel('Sample Size')
# axes[9].set_ylabel('W_FP')
# axes[9].legend(loc="upper right")

# # False Negative
# sns.tsplot(data=wfn_real_for_n.T, time=n, ci=[
#            68, 95], color='blue', condition='REAL', ax=axes[10])
# sns.tsplot(data=wfn_syn_for_n.T, time=n, ci=[
#            68, 95], color='orange', condition='SYN', ax=axes[10])
# axes[10].set_title('Sample Size vs Weighted False Negatives')
# axes[10].set_xlabel('Sample Size')
# axes[10].set_ylabel('W_FN')
# axes[10].legend(loc="upper right")

# Save results
figure.subplots_adjust(hspace=0.5)
figure.savefig('{0}[fmri_power_analysis]_[{1}_VS_{2}].pdf'.format(
    args.output_dir, args.dataset_1_label, args.dataset_2_label), format='pdf')
