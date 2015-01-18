'''
Implementation of some kernel methods.
'''

from numpy import linalg
import numpy as np
import kernels
import hselect


__all__ = ['sample_point_adaptive_weights',
           'update_dist',
           'kernel_regression',
           'set_bandwidth',
           'calc_dist',
           'kernel_density_estimate',
           'kernel_weights',
           'kernel_weights_non_radial']


def sample_point_adaptive_weights(X, kernel=kernels.gaussian, h='Silverman', alpha=0.5, dist=None):
    '''
    Calculates the sample point weights, using the Silverman adaptive kernel estimate.

    Parameters
    ----------
    X : np.array
        Coordinates of samples
    h : float or string
        Bandwidth of the kernel or method to use. Default 'Silverman'
    kernel : kernel class
        Kernel to use for the estimate. Default Gaussian
    alpha : float
        Sensivity parameter. Between 0 and 1. Setting to zero is equivalent to
        normal non-adaptive KDE. Setting to one is equivalent to nearest
        neighbor Adaptive KDE.
    dist : np.array
        Array of distances. This includes the distance from every point to every point.
        Prevents costly distance calculation done multple times.

    Returns
    -------
    log_lambda : np.array
        The array of weights.
    dist : np.array
        The array of calculated distances.
    bandwidth : float
        The computed bandwidth.
    '''

    # First calculate the normal bandwidth
    bandwidth = set_bandwidth(h, X.ravel())

    n = X.shape[0]

    # Calculate distances
    if dist is None:
        rep_X = np.repeat(X, n, 0)
        til_X = np.tile(X, (n, 1))

        dist = linalg.norm(rep_X - til_X, axis=1)
        dist = np.reshape(dist, (n, n))

    log_f_tilde = np.zeros(n)
    log_gmean = 0.0

    # Calculate the densities at each x location
    for i, col in enumerate(dist):
        log_f_tilde[i] = kernel_density_estimate(X[i], X, kernel, bandwidth, dist=col)
        log_gmean += log_f_tilde[i]

    log_gmean /= float(n)

    log_lambda = alpha * (log_gmean - log_f_tilde)

    return log_lambda, dist, bandwidth


def update_dist(new_point, xs, old_dist):
    '''
    Updates the distance matrix to include the distances of the old points to the
    new point.

    Parameters
    ----------
    new_point : np.array
        The new point to calculate all the distances to.
    xs : np.array
        The array of existing points.
    old_dist : np.array
        The old matrix of distances that needs to be augmented.

    Returns
    -------
    dist : np.array
        The augmented matrix of distances.
    '''
    new_row = calc_dist(new_point, xs)

    dist = np.vstack([old_dist, new_row[:-1]])
    dist = np.column_stack([dist, new_row])

    return dist


def kernel_regression(x_star, X, t, kernel=kernels.gaussian, h='SJ',
                      weights=None, dist=None):
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

    h = set_bandwidth(h, X.ravel())
    weights = kernel_weights(x_star, X, kernel, h=h, dist=dist)
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


def set_bandwidth(method, xs, weights=None):
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
    xs : array
        The x coordinates to use.
    weights : array (optional)
        The weights corresponding to the coordinates. Default None.
    '''
    try:
        h = float(method)
    except:
        bandwidth_func = {'SJ': hselect.hsj,
                          'Scott': hselect.hscott,
                          'Silverman': hselect.hsilverman}
        h = bandwidth_func[method](xs, weights)

    return h


def calc_dist(x_star, X):
    '''
    Calculate the distance of all points in X to x_star.
    '''
    return linalg.norm(x_star - X, axis=1)


def kernel_weights_non_radial(x_star, X, kernel, h='SJ', weights=None, dist=None):
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
    kweights = np.ones(X.shape[0])

    for d in xrange(dim):
        kweights *= kernel_weights(x_star, X[:, [d]], kernel, h, weights, dist)

    return kweights


def kernel_weights(x_star, X, kernel=kernels.gaussian, h='SJ',
                   weights=None, dist=None, ind_h=None, tree=None):
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
    ind_h : np.array
        Individual bandwidth terms, sometimes denoted lambda.
    tree : scipy.tree
        The KD tree to quickly compute involved points.

    Returns
    -------
    weights : array
        The array of weights for each training point
    '''

    h = set_bandwidth(h, X.ravel(), weights=weights)

    if h == 0.0:
        # If bandwidth is zero all weights should be infinity.
        # This effectively means that we need to perform simulations.
        return np.zeros(X.shape[0])

    if tree is None:
        if dist is None:
            u = linalg.norm(x_star - X, axis=1)
        else:
            u = dist

        if ind_h is None:
            return kernel(u / h) / h
        else:
            h_arr = h * ind_h
        return kernel(u / h_arr) / h_arr
    else:
        indices = tree.query_ball_point(x_star, 2 / h ** 2)

        u = np.ones(X.shape[0], dtype=float)

        if dist is None:
            u[indices] = linalg.norm(x_star - X[indices], axis=1)
        else:
            u[indices] = dist[indices]

        if ind_h is None:
            return kernel(u / h) / h
        else:
            h_arr = h * ind_h
            return kernel(u / h_arr) / h_arr


def kernel_density_estimate(x_star, X, kernel=kernels.gaussian, h='SJ',
                            weights=None, dist=None, nonradial=False):
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
    weights : np.array
        The array of weights associated with the data points.
        Default None.
    dist : np.array
        Array of distances to x_star for each data point. Avoids doubly
        calculating the distances, if you already have them.
        Default None.
    nonradial : bool
        Whether to use non radial kernel weights. i.e. multiplicative kernel.

    Returns
    -------
    log_estimate : float
        The log of estimated density at the probe location.
    log_conf: float
        The log of the confidence interval
    '''

    h = set_bandwidth(h, X.ravel(), weights=weights)

    if nonradial:
        kweights = kernel_weights_non_radial(x_star, X, kernel, h, weights, dist)
    else:
        kweights = kernel_weights(x_star, X, kernel, h, weights, dist)

    if weights is None:
        log_n = np.log(len(X))
        log_est = np.log(np.sum(kweights) + 1e-50) - log_n
    else:
        log_n = np.log(np.sum(weights))
        log_est = np.log(np.sum(weights * kweights) + 1e-50) - log_n

    log_R_gauss = - np.log(2) - 0.5 * np.log(np.pi)
    log_conf = np.log(1.96) + 0.5 * \
        (log_est + log_R_gauss - np.log(h) - log_n)

    return log_est#, log_conf
