import numpy as np

from scipy.stats import ttest_ind

def rejecting_voxels(d1, d2, alpha=0.05):
    two_sample_t_test_p_vals_by_voxel = np.zeros(d1.shape[1:])

    for i in range(two_sample_t_test_p_vals_by_voxel.shape[0]):
        for j in range(two_sample_t_test_p_vals_by_voxel.shape[1]):
            for k in range(two_sample_t_test_p_vals_by_voxel.shape[2]):
                d1_voxels = d1[:, i, j, k]
                d2_voxels = d2[:, i, j, k]
                two_sample_t_test_p_vals_by_voxel[i][j][k] = ttest_ind(d1_voxels, d2_voxels, equal_var=True).pvalue

    return fdr_correction(two_sample_t_test_p_vals_by_voxel, alpha=alpha)[0]

def rejection_mask_overlap(rejections, mask_rejections):
    num_overlap = 0
    total_mask_reject = 0
    total_either_reject = 0
    for i in range(rejections.shape[0]):
        for j in range(rejections.shape[1]):
            for k in range(rejections.shape[2]):
                rejection = rejections[i][j][k]
                mask_reject = mask_rejections[i][j][k]

                num_overlap += (rejection and mask_reject)
                total_mask_reject += mask_reject
                total_either_reject += (rejection or mask_reject)

    overlap_mask_ratio = float(num_overlap) / float(max(1, total_mask_reject))
    overlap_either_ratio = float(num_overlap) / float(max(1, total_either_reject))
    return overlap_mask_ratio, overlap_either_ratio

def power_calculations(d1, d2, n_1, n_2, overlap_mask, alpha=0.05, k=10**1):
    fdr_rejections = []
    overlap_mask_rejection_ratios = []
    overlap_either_rejection_ratios = []

    for br in range(k):
        d1_idx = np.random.randint(low=0, high=d1.shape[0], size=n_1)
        d2_idx = np.random.randint(low=0, high=d2.shape[0], size=n_2)

        d1_replicate = d1[d1_idx].squeeze()
        d2_replicate = d2[d2_idx].squeeze()

        fdr_reject_by_voxel = rejecting_voxels(d1_replicate, d2_replicate, alpha)
        fdr_reject = sum(fdr_reject_by_voxel) > 0  # reject if any voxel rejects
        fdr_rejections.append(fdr_reject)

        overlap_mask_ratio, overlap_either_ratio = rejection_mask_overlap(fdr_reject_by_voxel, overlap_mask)
        overlap_mask_rejection_ratios.append(overlap_mask_ratio)
        overlap_either_rejection_ratios.append(overlap_either_ratio)

    fdr_power = np.mean(fdr_rejections)
    return fdr_power, overlap_mask_rejection_ratios, overlap_either_rejection_ratios



# Authors: Josef Pktd and example from H Raja and rewrite from Vincent Davis
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# Code borrowed from statsmodels
#
# License: BSD (3-clause)
def _ecdf(x):
    """No frills empirical cdf used in fdrcorrection."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def fdr_correction(pvals, alpha=0.05, method='indep'):
    """P-value correction with False Discovery Rate (FDR).

    Correction for multiple comparison using FDR.

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : 'indep' | 'negcorr'
        If 'indep' it implements Benjamini/Hochberg for independent or if
        'negcorr' it corresponds to Benjamini/Yekutieli.

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----
    Reference:
    Genovese CR, Lazar NA, Nichols T.
    Thresholding of statistical maps in functional neuroimaging using the false
    discovery rate. Neuroimage. 2002 Apr;15(4):870-8.
    """
    pvals = np.asarray(pvals)
    shape_init = pvals.shape
    pvals = pvals.ravel()

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = pvals[pvals_sortind]
    sortrevind = pvals_sortind.argsort()

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1. / np.arange(1, len(pvals_sorted) + 1))
        ecdffactor = _ecdf(pvals_sorted) / cm
    else:
        raise ValueError("Method should be 'indep' and 'negcorr'")

    reject = pvals_sorted < (ecdffactor * alpha)
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
    else:
        rejectmax = 0
    reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1.0] = 1.0
    pvals_corrected = pvals_corrected[sortrevind].reshape(shape_init)
    reject = reject[sortrevind].reshape(shape_init)
    return reject, pvals_corrected


def bonferroni_correction(pval, alpha=0.05):
    """P-value correction with Bonferroni method.

    Parameters
    ----------
    pval : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate

    Returns
    -------
    reject : array, bool
        True if a hypothesis is rejected, False if not
    pval_corrected : array
        pvalues adjusted for multiple hypothesis testing to limit FDR

    """
    pval = np.asarray(pval)
    pval_corrected = pval * float(pval.size)
    reject = pval_corrected < alpha
    return reject, pval_corrected
