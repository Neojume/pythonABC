'''
Implements the locally weighted regression in multiple dimensions.
'''

import numpy as np
from scipy import linalg
import kernels
import kernel_methods as km


def lowessNd(x_star, x, y, kernel=kernels.tricube, bandwidth=2. / 3., radial=False, dist=None):
    '''
    Performs locally weighted linear regression in multiple dimensions.
    For more information see [1]_ and [2]_.

    Parameters
    ----------
    x_star : array
        Location of estimate.
    x : array
        Array of x-coordinates of data. Shape NxM where N is number of
        points and M the dimensionality.
    y : array
        Array of y-coordinates of data.
    kernel : kernel function
        The kernel to weigh the data with. Default is tricube.
    bandwidth : float
        The bandwidth to use. Default 2 / 3.
    radial : bool (optional)
        Whether to use a radial kernel. Default False.

    References
    ----------
    .. [1] William S. Cleveland: "Robust locally weighted regression and smoothing
       scatterplots", Journal of the American Statistical Association, December 1979,
       volume 74, number 368, pp. 829-836.

    .. [2] William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
       approach to regression analysis by local fitting", Journal of the American
       Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
    '''

    n = x.shape[0]
    dim = x.shape[1]
    if dist is None:
        dist = np.linalg.norm(x - x_star, axis=1)
    d = np.array(dist, ndmin=2).T
    r = min(np.floor(n / 2), int(np.ceil(bandwidth * n)))
    h = np.sort(d)[r]
    w = kernel(d / h)

    # TODO: Fix radial
    #if radial:
    #    w = km.kernel_weights(x_star, x, kernel, bandwidth)
    #else:
    #    w = km.kernel_weights_non_radial(x_star, x, kernel, bandwidth)

    x_ext = np.hstack([np.ones((n, 1)), x])

    rng = range(dim + 1)

    b = np.array([np.sum(y * w * x_ext[:, [i]]) for i in rng])
    A = np.array([[np.sum(w * x_ext[:, [i]] * x_ext[:, [j]]) for i in rng]
                  for j in rng])
    try:
        beta = linalg.solve(A, b)
    except:
        # If system of equations is singular: assume horizontal line
        beta = np.zeros(dim + 1)

    return beta, dist
