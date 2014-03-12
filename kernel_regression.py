import numpy as np
from random import uniform, gauss
import matplotlib.pyplot as plt
from numpy import linalg, log
from collections import defaultdict
import kernels


def adaptive_kernel_regression(x_star, X, t, h, kernel=kernels.gaussian,
                               alpha=0):
    '''
    Returns the kernel density estimate at x_star using the given kernel and
    bandwidth on the given data.

    Parameters
    ----------
    x_star : estimation location

    X : coordinates of samples

    t : sample values

    kernel : kernel to use for the estimate, default epanechnikov

    bandwidth : bandwidth of the kernel, default 1
    '''

    # Calculate adaptive bandwidths
    bandwidth = defaultdict(lambda: alpha)
    for i, x1 in enumerate(X):
        for x2 in X:
            u = linalg.norm(x1 - x2) / h
            bandwidth[i] += (kernel(u) / h) * (x1 - x2) ** 2

    weights = np.zeros(len(X))
    for i, x in enumerate(X):
        u = linalg.norm(x_star - x) / bandwidth[i]
        weights[i] = kernel(u) / bandwidth[i]

    weighted = weights * t

    N = np.log(np.sum(weights))

    mean = np.sum(weighted) / np.exp(N)

    sigma = np.log(np.sum(weights * np.square(mean - t)))
    std = 0.5 * (sigma - N)
    return mean, np.exp(std), N


def MLCV(h, X, kernel):
    ans = 0
    n = len(X)

    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            if i == j:
                continue

            u = linalg.norm(x2 - x1) / h
            ans += np.log(kernel(u))
        ans -= - np.log((n - 1) * h)

    # NOTE: Return minus the answer, because scipi only minimizes
    return ans


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

    kernel : kernel to use for the estimate, default Gaussian

    h : bandwidth of the kernel, default MLVC estimate
    '''

    u = linalg.norm(x_star - X, axis=1) / h
    weights = kernel(u) / h
    weighted = t.T * weights

    N = np.log(np.sum(weights))
    mean = np.sum(weighted) / np.exp(N)

    summ = log(np.sum(np.square(mean - t).T * weights))
    log_std = 0.5 * (summ - N)

    # TODO: currently hardcoded for Gaussian: make more general
    # NOTE: currently zero bias assumed
    log_conf = log(1.96) + 0.5 * (log(2 * np.pi) + log_std - N - log(h))
    return mean, np.exp(log_std), np.exp(log_conf), np.exp(N)


def kernel_regression_old(x_star, X, t, h, kernel=kernels.gaussian):

    t = np.array(t)
    weights = np.zeros(len(X))
    for i, x in enumerate(np.array(X)):
        u = linalg.norm(x_star - x) / h
        weights[i] = kernel(u) / h

    weighted = weights * t

    N = np.sum(weights)
    mean = np.sum(weighted) / N

    std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) / N)
    # TODO: for now hardcoded for Gaussian: replace with more general code
    # NOTE: zero bias assumed
    conf = 1.96 * np.sqrt((2.0 * np.pi * std) / (N * h))
    return mean, std, conf, N


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


def real_func(x):
    return 20 * np.sin(x / 1.2) + 2 * x

if __name__ == '__main__':

    num_samples = 300
    noise_level = 2
    limits = [0.0, 4 * np.pi]

    x_coords = [uniform(limits[0], limits[1]) for i in range(num_samples)]

    # So that there is a gap in between

    #d = limits[1] - limits[0]
    #x_coords = [uniform(limits[0], limits[0] + d / 4.0)
    #            for i in range(num_samples / 2)]
    #x_coords += [uniform(limits[1] - d / 4.0, limits[1])
    #             for i in range(num_samples / 2)]

    gap = biggest_gap(x_coords, limits)

    noise = [gauss(0.0, noise_level) for i in range(num_samples)]
    X = np.array(x_coords, ndmin=2).T
    noise_arr = np.array(noise, ndmin=2).T
    t = real_func(X) + noise_arr

    precision = 200

    test_range = np.linspace(limits[0], limits[1], precision)
    bandwidth_range = np.linspace(gap, limits[1] - limits[0], precision)

    errors = []

    bandwidth = 0.3

    # Reset the means and stds
    means = np.zeros(precision)
    stds = np.zeros(precision)
    confs = np.zeros(precision)
    Ns = np.zeros(precision)

    # Compute the kernel density estimates
    for i, val in enumerate(test_range):
        means[i], stds[i], confs[i], Ns[i] = \
            kernel_regression(np.array(val), X, t, bandwidth)

    # Compute the error and add it to the list
    error = MSE(means, real_func(test_range))
    errors.append(error)

    # Plot the fit
    ax1 = plt.subplot(211)
    ax1.set_ylabel('KRE')
    plt.plot(test_range, real_func(test_range), color='r')
    plt.plot(test_range, means, color='y')
    plt.scatter(X, t)

    plt.fill_between(test_range,
                     (means - stds / np.sqrt(Ns)),
                     (means + stds / np.sqrt(Ns)),
                     color=[0.7, 0.3, 0.3, 0.5])
    plt.fill_between(test_range,
                     (means - 2 * stds),
                     (means + 2 * stds),
                     color=[0.7, 0.7, 0.7, 0.7])
    plt.scatter(X.flat, t.flat)
    plt.xlim(limits[0], limits[1])

    plt.subplot(212, sharex=ax1)
    plt.plot(test_range, confs, label='confs')
    plt.plot(test_range, stds, label='stds')
    plt.legend()

    plt.show()
