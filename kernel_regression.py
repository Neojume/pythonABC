# Used libraries
from numpy import linalg
import numpy as np

# Own imports
import kernels


def adaptive_kernel_regression(x_star, X, t, h, kernel=kernels.gaussian,
                               alpha=0):
    '''
    Returns the kernel density estimate at x_star using the given kernel and
    an adaptive bandwidth on the given data.

    Parameters
    ----------
    x_star : np.array
        Estimate location

    X : np.array
        Coordinates of samples

    t : np.array
        Sample values

    h : float
        Bandwidth of the kernel

    kernel : kernel class
        kernel to use for the estimate, default Gaussian
    '''

    # Calculate adaptive bandwidths
    #bandwidth = defaultdict(lambda: alpha)
    #for i, x1 in enumerate(X):
    #    for x2 in X:
    #        u = linalg.norm(x1 - x2) / h
    #        bandwidth[i] += (kernel(u) / h) * (x1 - x2) ** 2

    n = len(X)
    rep_X = np.repeat(X, n, 0)
    til_X = np.tile(X, (n, 1))
    u = linalg.norm(rep_X - til_X, axis=1)
    r = kernel(u) * u ** 2
    bandwidth = np.sum(np.reshape(r, (n, n)), axis=1)

    #weights = np.zeros(len(X))
    #for i, x in enumerate(X):
    #    u = linalg.norm(x_star - x) / bandwidth[i]
    #    weights[i] = kernel(u) / bandwidth[i]

    u = linalg.norm(x_star - X, axis=1) / bandwidth
    weights = kernel(u) / bandwidth
    weighted = t.T * weights

    N = np.log(np.sum(weights))
    mean = np.sum(weighted) / np.exp(N)

    summ = np.log(np.sum(np.square(mean - t).T * weights))
    log_std = 0.5 * (summ - N)
    weighted = weights * t

    #N = np.log(np.sum(weights))

    #mean = np.sum(weighted) / np.exp(N)

    #sigma = np.log(np.sum(weights * np.square(mean - t)))
    #std = 0.5 * (sigma - N)
    return mean, np.exp(log_std), 0, N


def kernel_regression(x_star, X, t, h, kernel=kernels.gaussian):
    '''
    Returns the kernel regression estimate at x_star using the given kernel and
    bandwidth on the given data.

    Parameters
    ----------
    x_star : np.array
        Estimate location

    X : np.array
        Coordinates of samples

    t : np.array
        Sample values

    h : float
        Bandwidth of the kernel

    kernel : kernel class
        kernel to use for the estimate, default Gaussian
    '''

    u = linalg.norm(x_star - X, axis=1) / h
    weights = kernel(u) / h
    weighted = t.T * weights

    N = np.log(np.sum(weights))
    mean = np.sum(weighted) / np.exp(N)

    summ = np.log(np.sum(np.square(mean - t).T * weights))
    log_std = 0.5 * (summ - N)

    # TODO: currently hardcoded for Gaussian: make more general
    # NOTE: currently zero bias assumed
    log_conf = np.log(1.96) + 0.5 * \
        (np.log(2 * np.pi) + log_std - N - np.log(h))
    return mean, np.exp(log_std), np.exp(log_conf), np.exp(N)


def MSE(predictions, targets):
    '''
    Returns the mean squared error.
    '''
    return np.mean((predictions - targets) ** 2)


def biggest_gap(x_coords, limits):
    gap = 0.0
    prev = limits[0]
    for x in sorted(x_coords + [limits[1]]):
        diff = x - prev
        if diff > gap:
            gap = diff
        prev = x
    return gap
