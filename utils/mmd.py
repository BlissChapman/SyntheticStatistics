import math
import numpy as np

def mmd(X, Y, sigma=None, alpha=0.05, k=100):
    # Compute MMD stat on original data split:
    mmd_stat = mmd_test(X, Y, sigma=sigma)[1]

    # Compute distribution of MMD stat via permutation test:
    Z = np.concatenate((X, Y), axis=0)
    mmd_permutations = np.zeros(k)

    for k_i in range(k):
        split_idx = X.shape[0]
        X_k = Z[:split_idx]
        Y_k = Z[split_idx:]

        mmd_stat_k = mmd_test(X_k, Y_k, sigma=sigma)[1]
        mmd_permutations[k_i] = mmd_stat_k

        np.random.shuffle(Z)

    # Reject the null if the MMD stat of original split
    #  is significant under the distribution of the MMD statistic:
    mmd_reject = False
    if abs(mmd_stat) > abs(np.percentile(mmd_permutations, 1.0-alpha)):
        mmd_reject = True

    return mmd_reject


def grbf(x1, x2, sigma):
    '''Calculates the Gaussian radial base function kernel'''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    return np.exp(-1 * h / (2 * pow(sigma, 2)))


def kernelwidth(x1, x2):
    '''Function to estimate the sigma parameter

       The RBF kernel width sigma is computed according to a rule of thumb:

       Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
       in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
       and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    mdist = np.median([i for i in h.flat if i])

    sigma = math.sqrt(mdist / 2.0)
    if not sigma:
        sigma = 1

    return sigma


def mmd_test(x1, x2, sigma=None, verbose=False):
    '''Calculates the unbiased mmd from two arrays x1 and x2

    sigma: the parameter for grbf. If None sigma is estimated

    Returns (sigma, mmd)

    '''
    if x1.size != x2.size:
        raise ValueError('Arrays should have an equal amount of instances')

    # Number of instances
    m, nfeatures = x1.shape

    # Calculate sigma
    if sigma is None:
        sigma = kernelwidth(x1, x2)

    # Calculate the kernels
    Kxx = grbf(x1, x1, sigma)
    Kyy = grbf(x2, x2, sigma)
    s = Kxx + Kyy
    del Kxx, Kyy

    Kxy = grbf(x1, x2, sigma)
    s = s - Kxy - Kxy
    del Kxy

    # For unbiased estimator: subtract diagonal
    s = s - np.diag(s.diagonal())
    value = np.sum(s) / (m * (m - 1))

    return sigma, value
