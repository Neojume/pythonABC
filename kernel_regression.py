from numpy import linalg
import numpy as np
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

    # TODO: Fix this
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
        Estimate location.
        Of length M, where M is the dimensionality of x
        Note: 1D
    X : np.array
        Coordinates of samples.
        N x M, where M is the dimensionality of x, N the number of samples
    t : np.array
        Sample values.
        N x 1, where N is the number of samples
    h : float
        Bandwidth of the kernel
    kernel : kernel class
        kernel to use for the estimate, default Gaussian

    Returns
    -------
    mean, std, conf, N : tuple
        The mean, standard deviation and confidence at the probe location.
        `N` is the weighted number of samples used.
    '''

    u = linalg.norm(x_star - X, axis=1) / h
    weights = kernel(u) / h
    weighted = t.T * weights

    N = np.log(np.sum(weights))
    mean = np.sum(weighted) / np.exp(N)

    summ = np.log(np.sum(np.square(mean - t).T * weights))
    log_std = 0.5 * (summ - N)

    # x_m = X - mean
    cov = 0  # np.dot(weights * x_m, x_m.T) / float(X.shape[-1] - 1)

    # TODO: currently hardcoded for Gaussian: make more general
    # NOTE: currently zero bias assumed
    log_conf = np.log(1.96) + 0.5 * \
        (np.log(2 * np.pi) + log_std - N - np.log(h))
    return mean, np.exp(log_std), np.exp(log_conf), np.exp(N), cov


def doubly_kernel_estimate(x_star, y_star, X, t,
                           kernel_x=kernels.gaussian, h_x=1,
                           kernel_y=kernels.gaussian, h_y=1):
    '''
    Returns the logarithm of the density estimate of y_star at location
    x_star.

    Parameters
    ----------
    x_star : np.array
        Estimate location.
        Of length M, where M is the dimensionality of x
        Note: 1D
    y_star : float
        The value to calculate the density of.
    X : np.array
        Coordinates of samples.
        N x M, where M is the dimensionality of x, N the number of samples
    t : np.array
        Sample values.
        N x 1, where N is the number of samples
    kernel_x : kernel class
        kernel to use for the estimate in the x direction, default Gaussian
    h_x : float
        Bandwidth of the x-kernel
    kernel_y : kernel class
        kernel to use for the estimate of the density function, default Gaussian
    h_y : float
        Bandwidth of the y-kernel

    Returns
    -------
    log_estimate : float
        The logarithm of the density estimate.
    '''

    # TODO: Make it work when more x or y are queried
    x_star = problem.rng[0] + 0.75 * problem.rng[1]
    u_x = np.linalg.norm(x_star - xs, axis=1) / h_x
    weights_x = kernel(u_x) / h_x

    #weights_x = np.array(weights_x, ndmin=2).T

    u_y = np.linalg.norm(y_star - ts, axis=1) / h_x
    weights_y = kernel(u_y) / h_y
    weights_y *= weights_x

    return sum(weights_y) / float(sum(weights_x))



def kernel_density_estimate(x_star, X, t, h, kernel=kernels.gaussian):
    '''
    Returns the kernel density estimate at x_star using the given kernel and
    bandwidth on the given data.

    Parameters
    ----------
    x_star : np.array
        Estimate location.
        Of length M, where M is the dimensionality of x
        Note: 1D
    X : np.array
        Coordinates of samples.
        N x M, where M is the dimensionality of x, N the number of samples
    t : np.array
        Sample values.
        N x 1, where N is the number of samples
    h : float
        Bandwidth of the kernel
    kernel : kernel class
        kernel to use for the estimate, default Gaussian

    Returns
    -------
    log_estimate : float
        The log of estimated density at the probe location.
    '''
    u = linalg.norm(x_star - X, axis=1) / h
    weights = kernel(u) / h

    return  np.log(np.sum(weights))


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
