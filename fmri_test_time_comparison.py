import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil

from brainpedia.brainpedia import Brainpedia
from brainpedia.fmri_processing import invert_preprocessor_scaling
from utils.multiple_comparison import rejecting_voxels, power_calculations
from nilearn import plotting
from utils.sampling import *


# Parse arguments
parser = argparse.ArgumentParser(description="Compare classical two sample t test to non-parametric tests for real and synthetic fMRI brain imaging datasets.")
parser.add_argument('real_dataset_dir', help='the directory containing the first real fMRI dataset')
parser.add_argument('real_dataset_cache_dir', help='the directory to use as a cache for real_dataset preprocessing')
parser.add_argument('syn_dataset_dir', help='the directory containing the synthetic fMRI dataset generated from a model trained on real_dataset')
parser.add_argument('syn_dataset_cache_dir', help='the directory to use as a cache for syn_dataset preprocessing')
parser.add_argument('output_dir', help='the directory to save comparison results')
args = parser.parse_args()

# Setup output directory
shutil.rmtree(args.output_dir, ignore_errors=True)
os.makedirs(args.output_dir)

# Load datasets
DOWNSAMPLE_SCALE = 0.25
MULTI_TAG_LABEL_ENCODING = False

real_dataset_brainpedia = Brainpedia(data_dirs=[args.real_dataset_dir],
                                     cache_dir=args.real_dataset_cache_dir,
                                     scale=DOWNSAMPLE_SCALE,
                                     multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)
syn_dataset_brainpedia = Brainpedia(data_dirs=[args.syn_dataset_dir],
                                    cache_dir=args.syn_dataset_cache_dir,
                                    scale=DOWNSAMPLE_SCALE,
                                    multi_tag_label_encoding=MULTI_TAG_LABEL_ENCODING)

real_dataset, real_dataset_tags = real_dataset_brainpedia.all_data()
syn_dataset, syn_dataset_tags = syn_dataset_brainpedia.all_data()

# Filter real and synthetic datasets into visual and non-visual datasets
real_dataset_non_visual = []
syn_dataset_non_visual = []

real_dataset_visual = []
syn_dataset_visual = []

for (data, tag) in zip(real_dataset, real_dataset_tags):
    if 'visual' in real_dataset_brainpedia.preprocessor.decode_label(tag):
        real_dataset_visual.append(data)
    else:
        real_dataset_non_visual.append(data)

for (data, tag) in zip(syn_dataset, syn_dataset_tags):
    if 'visual' in syn_dataset_brainpedia.preprocessor.decode_label(tag):
        syn_dataset_visual.append(data)
    else:
        syn_dataset_non_visual.append(data)

# Trim real datasets to the same length
real_dataset_length = min(len(real_dataset_visual), len(real_dataset_non_visual))
real_dataset_visual = np.array(real_dataset_visual[:real_dataset_length])
real_dataset_non_visual = np.array(real_dataset_non_visual[:real_dataset_length])

# Trim synthetic datasets to the same length
syn_dataset_length = min(len(syn_dataset_visual), len(syn_dataset_non_visual))
syn_dataset_visual = np.array(syn_dataset_visual[:syn_dataset_length])
syn_dataset_non_visual = np.array(syn_dataset_non_visual[:syn_dataset_length])

# Plot examples from datasets
real_non_visual_brain_img = invert_preprocessor_scaling(real_dataset_non_visual[0].squeeze(), real_dataset_brainpedia.preprocessor)
real_visual_brain_img = invert_preprocessor_scaling(real_dataset_visual[0].squeeze(), real_dataset_brainpedia.preprocessor)
syn_non_visual_brain_img = invert_preprocessor_scaling(syn_dataset_non_visual[2].squeeze(), syn_dataset_brainpedia.preprocessor)
syn_visual_brain_img = invert_preprocessor_scaling(syn_dataset_visual[0].squeeze(), syn_dataset_brainpedia.preprocessor)

figure, axes = plt.subplots(nrows=9, ncols=1, figsize=(15, 40))
plotting.plot_glass_brain(real_non_visual_brain_img, threshold='auto', title="[REAL NON-VISUAL]", axes=axes[0])
plotting.plot_glass_brain(syn_non_visual_brain_img, threshold='auto', title="[SYN NON-VISUAL]", axes=axes[1])
plotting.plot_glass_brain(real_visual_brain_img, threshold='auto', title="[REAL VISUAL]", axes=axes[2])
plotting.plot_glass_brain(syn_visual_brain_img, threshold='auto', title="[SYN VISUAL]", axes=axes[3])

# Compute voxels that are statistically significant indicators of non-visual vs visual
visual_vs_non_visual_rejecting_voxels = rejecting_voxels(real_dataset_visual.squeeze(), real_dataset_non_visual.squeeze())

# Compute power for various n
k = 10*1
n = np.geomspace(2, syn_dataset_length, num=25)

fdr_test_power_for_n = np.zeros(len(n))
syn_fdr_test_power_for_n = np.zeros(len(n))

tp_ratios_syn_for_n = np.zeros((len(n), k))
tn_ratios_syn_for_n = np.zeros((len(n), k))
fp_ratios_syn_for_n = np.zeros((len(n), k))
fn_ratios_syn_for_n = np.zeros((len(n), k))

tp_ratios_real_for_n = np.zeros((len(n), k))
tn_ratios_real_for_n = np.zeros((len(n), k))
fp_ratios_real_for_n = np.zeros((len(n), k))
fn_ratios_real_for_n = np.zeros((len(n), k))

for i in range(len(n)):
    # Determine sample sizes to draw from synthetic and real datasets
    # Note: there is limited real data. When there is none left, simply use the
    #   max available amount of data.
    syn_n = int(n[i])
    real_n = min(real_dataset_non_visual.shape[0], int(n[i]))
    fdr_syn_power, tp_ratios_syn, tn_ratios_syn, fp_ratios_syn, fn_ratios_syn = power_calculations(syn_dataset_non_visual, syn_dataset_visual, syn_n, syn_n, visual_vs_non_visual_rejecting_voxels, k=k)
    fdr_real_power, tp_ratios_real, tn_ratios_real, fp_ratios_real, fn_ratios_real = power_calculations(real_dataset_non_visual, real_dataset_visual, real_n, real_n, visual_vs_non_visual_rejecting_voxels, k=k)

    fdr_test_power_for_n[i] = fdr_real_power
    syn_fdr_test_power_for_n[i] = fdr_syn_power

    tp_ratios_syn_for_n[i][:] = tp_ratios_syn[:]
    tn_ratios_syn_for_n[i][:] = tn_ratios_syn[:]
    fp_ratios_syn_for_n[i][:] = fp_ratios_syn[:]
    fn_ratios_syn_for_n[i][:] = fn_ratios_syn[:]

    tp_ratios_real_for_n[i][:] = tp_ratios_real[:]
    tn_ratios_real_for_n[i][:] = tn_ratios_real[:]
    fp_ratios_real_for_n[i][:] = fp_ratios_real[:]
    fn_ratios_real_for_n[i][:] = fn_ratios_real[:]

    print("PERCENT COMPLETE: {0:.2f}%\r".format(100 * float(i) / float(len(n))), end='')

# Plot curve of n vs power
axes[4].plot(n, fdr_test_power_for_n, label='FDR Test Real')
axes[4].plot(n, syn_fdr_test_power_for_n, label='FDR Test Syn')
axes[4].set_title('Sample Size vs Power')
axes[4].set_xlabel('Sample Size')
axes[4].set_ylabel('Power')
axes[4].set_ylim([-0.1, 1.1])
axes[4].legend(loc="upper right")

# Plot curve of n vs rejection overlaps
# True Positive
sns.tsplot(data=tp_ratios_real_for_n.T, time=n, ci=[68, 95], color='blue', condition='REAL', ax=axes[5])
sns.tsplot(data=tp_ratios_syn_for_n.T, time=n, ci=[68, 95], color='orange', condition='SYN', ax=axes[5])
axes[5].set_title('Sample Size vs True Positives')
axes[5].set_xlabel('Sample Size')
axes[5].set_ylabel('TP / |ROI Reject|')
axes[5].set_ylim([-0.1, 1.1])
axes[5].legend(loc="upper right")

# True Negative
sns.tsplot(data=tn_ratios_real_for_n.T, time=n, ci=[68, 95], color='blue', condition='REAL', ax=axes[6])
sns.tsplot(data=tn_ratios_syn_for_n.T, time=n, ci=[68, 95], color='orange', condition='SYN', ax=axes[6])
axes[6].set_title('Sample Size vs True Negatives')
axes[6].set_xlabel('Sample Size')
axes[6].set_ylabel('TN / |ROI Accept|')
axes[6].set_ylim([-0.1, 1.1])
axes[6].legend(loc="upper right")

# False Positive
sns.tsplot(data=fp_ratios_real_for_n.T, time=n, ci=[68, 95], color='blue', condition='REAL', ax=axes[7])
sns.tsplot(data=fp_ratios_syn_for_n.T, time=n, ci=[68, 95], color='orange', condition='SYN', ax=axes[7])
axes[7].set_title('Sample Size vs False Positives')
axes[7].set_xlabel('Sample Size')
axes[7].set_ylabel('FP / |ROI Accept|')
axes[7].set_ylim([-0.1, 1.1])
axes[7].legend(loc="upper right")

# False Negative
sns.tsplot(data=fn_ratios_real_for_n.T, time=n, ci=[68, 95], color='blue', condition='REAL', ax=axes[8])
sns.tsplot(data=fn_ratios_syn_for_n.T, time=n, ci=[68, 95], color='orange', condition='SYN', ax=axes[8])
axes[8].set_title('Sample Size vs False Negatives')
axes[8].set_xlabel('Sample Size')
axes[8].set_ylabel('FN / |ROI Reject|')
axes[8].set_ylim([-0.1, 1.1])
axes[8].legend(loc="upper right")

# Save results
figure.subplots_adjust(hspace=0.5)
figure.savefig('{0}sample_size_vs_power.png'.format(args.output_dir))
