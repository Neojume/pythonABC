'''
Implementation of some kernel methods.
'''

from numpy import linalg
import numpy as np
import kernels
import hselect


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


def kernel_regression(x_star, X, t, kernel=kernels.gaussian, h='SJ'):
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
    h : float or string
        Bandwidth estimation method. If float, that is used as bandwidth.
        Default `SJ`, which is the Sheather-Jones plug-in estimate.
    kernel : kernel class
        kernel to use for the estimate, default Gaussian

    Returns
    -------
    mean, std, conf, N : tuple
        The mean, standard deviation and confidence at the probe location.
        `N` is the weighted number of samples used.
    '''

    weights = kernel_weights(x_star, X, kernel, h)
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


def set_bandwidth(method, xs):
    '''
    Sets the bandwidth using the given method.

    Arguments
    ---------
    method : float or string
        The method to set the bandwidth with. If `float` that will
        be used as the bandwidth. If string:
        `SJ` = Sheather-Jones plug-in estimate.
        `Scott` = Scotts rule of thumb.
        `Silverman` = Silvermans rule of thumb.
    '''
    try:
        h = float(method)
    except:
        bandwidth_func = {'SJ': hselect.hsj,
                          'Scott': hselect.hscott,
                          'Silverman': hselect.hsilverman}
        h = bandwidth_func[method](xs)

    return h


def doubly_kernel_estimate(x_star, y_star, X, t,
                           kernel_x=kernels.gaussian, h_x='SJ',
                           kernel_y=kernels.gaussian, h_y='SJ'):
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
    h_x : float or string
        Bandwidth of the x-kernel
    kernel_y : kernel class
        kernel to use for the estimate of the density function.
        Default Gaussian.
    h_y : float float or string
        Bandwidth of the y-kernel

    Returns
    -------
    log_estimate : float
        The logarithm of the density estimate.
    '''

    h_x = set_bandwidth(h_x, X.ravel())
    h_y = set_bandwidth(h_y, t.ravel())

    # TODO: Make it work when more x or y are queried
    weights_x = kernel_weights(x_star, X, kernel_x, h_x)
    weights_y = kernel_weights(y_star, t, kernel_y, h_y)
    weights_y *= weights_x

    return np.log(sum(weights_y)) - np.log(sum(weights_x))


def kernel_weights_non_radial(x_star, X, kernel, h='SJ'):
    '''
    Returns the non radial kernel-weights for the data points given the x-star.
    This means that each dimension has its own bandwidth.

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
    kernel : kernel class
        kernel to use for the estimate, default Gaussian
    h : float or string
        Bandwidth of the kernel

    Returns
    -------
    weights : array
        The array of weights for each training point
    '''

    dim = X.shape[1]
    weights = np.ones(X.shape[0])

    for d in xrange(dim):
        weights *= kernel_weights(x_star[d], X[:, [d]], kernel, h)

    return weights


def kernel_weights(x_star, X, kernel=kernels.gaussian, h='SJ'):
    '''
    Returns the kernel-weights for the data points given the x-star.

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
    kernel : kernel class
        kernel to use for the estimate, default Gaussian
    h : float or string
        Bandwidth of the kernel

    Returns
    -------
    weights : array
        The array of weights for each training point
    '''

    h = set_bandwidth(h, X.ravel())

    u = linalg.norm(x_star - X, axis=1) / h
    return kernel(u) / h


def kernel_density_estimate(x_star, X, kernel=kernels.gaussian, h='SJ'):
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

    weights = kernel_weights(x_star, X, kernel, h)

    return np.log(np.sum(weights)) - np.log(len(X))
