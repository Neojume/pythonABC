from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import kernel_methods as km
import kernels


__all__ = ['get_bootstrap_ids',
           'get_bootstrap',
           'get_weighted_bootstrap',
           'conditional_error',
           'logsumexp',
           'peakdet',
           'lowessNd',
           'plot_statistics',
           'plot_samples']


def get_bootstrap_ids(N, weights=None):
    '''
    Provides bootstrap indices.

    Parameters
    ----------
    N : integer
        Number of samples.
    weights : array
        Array of weights associated with each sample. Default `None`.

    Returns
    -------
    ids : array
        Array of indices for the bootstrap sample.
    '''

    if weights is None:
        ids = np.random.random_integers(0, N - 1, N)

        return ids
    else:
        target = float(sum(weights))

        c = np.cumsum(weights)
        ids = []

        w = 0.0
        while w < target:
            sid = np.searchsorted(c, np.random.uniform(0, target))
            ids.append(sid)
            w += weights[sid]

        return np.array(ids)


def get_bootstrap(values, weights=None):
    '''
    Provides a bootstrap sample of the given samples.

    Parameters
    ----------
    values : array
        Array of samples.
    weights : array
        Array of weights associated with each sample. Default `None`.
    '''

    values = np.asarray(values)

    if weights is None:
        ids = np.random.random_integers(0, len(values) - 1, len(values))

        return values[ids]
    else:
        target = float(sum(weights))

        c = np.cumsum(weights)
        samples = []

        w = 0.0
        while w < target:
            sid = np.searchsorted(c, np.random.uniform(0, target))
            samples.append(values[sid])
            w += weights[sid]

        return np.array(samples)


def get_weighted_bootstrap(values, weights, num_samples):
    '''
    Get a bootstrap sample of the given values, each value picked with the
    given weight.
    '''

    values = np.asarray(values)
    c = np.cumsum(weights)
    r = np.random.uniform(0, sum(weights), num_samples)
    return values[np.searchsorted(c, r)]


def conditional_error(alphas, u, tau, M):
    '''
    Computes the conditional error as given by equation 13 and 14.
    '''
    if u <= tau:
        return float(np.sum(alphas < u)) / M
    else:
        return float(np.sum(alphas >= u)) / M


def logsumexp(x, dim=0):
    '''
    Compute log(sum(exp(x))) in numerically stable way.
    '''

    if dim == 0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x - xmax).sum(0))
    # TODO: support other dimensions
    # NOTE: Newaxis is not defined
    #elif dim == 1:
    #    xmax = x.max(1)
    #    return xmax + np.log(np.exp(x - xmax[:, newaxis]).sum(1))
    else:
        raise 'dim ' + str(dim) + 'not supported'


def plot_samples(problem, samples):
    '''
    Plots the histograms and scatterplots of the different dimensions
    of the samples.
    '''

    asamples = np.array(samples, ndmin=2)
    print asamples.shape
    num_samples = asamples.shape[0]
    pars = problem.simulator_args
    num_pars = len(pars)
    print num_pars
    has_true_vals = problem.true_args is not None
    has_true_post = problem.true_posterior is not None

    par_values = dict()

    xsmin = dict()
    xsmax = dict()
    ysmin = dict()
    ysmax = dict()

    for i, par in enumerate(pars):
        par_values[par] = asamples[:, i]
        xsmin[i] = []
        xsmax[i] = []
        ysmin[i] = []
        ysmax[i] = []

    axes = dict()

    plt.figure(0)
    for i, par1 in enumerate(pars):
        for j, par2 in enumerate(pars):
            axes[i, j] = plt.subplot(num_pars, num_pars, num_pars * i + j + 1)

    for i, par1 in enumerate(pars):
        for j, par2 in enumerate(pars):
            ax = axes[i, j]

            if i == j:
                # Same parameter, so make a histogram
                ax.hist(par_values[par1], 50, normed=True)
                if has_true_vals:
                    ax.axvline(problem.true_args[i], color='r')

                if has_true_post:
                    lim = problem.true_posterior_rng
                    rng = np.linspace(lim[0], lim[1], 100)
                    ax.plot(rng, problem.true_posterior.pdf(rng, *problem.true_posterior_args))
            else:
                # Different parameters, make scatterplot
                ax.scatter(par_values[par2], par_values[par1], c=range(num_samples), cmap=plt.get_cmap('autumn'))
                if has_true_vals:
                    ax.scatter(problem.true_args[j], problem.true_args[i],
                               color='r')

            # Set the ticks and labels to the right positions and values
            if j == 0:
                ax.set_ylabel(pars[i])
            elif j == num_pars - 1:
                ax.get_yaxis().tick_right()
                ax.set_ylabel(pars[i])
                ax.get_yaxis().set_label_position('right')
            else:
                ax.get_yaxis().set_ticklabels([])

            if i == 0:
                ax.get_xaxis().tick_top()
                ax.get_xaxis().set_label_position('top')
                ax.set_xlabel(pars[j])
            elif i == num_pars - 1:
                ax.set_xlabel(pars[j])
            else:
                ax.get_xaxis().set_ticklabels([])

    if num_pars > 1:
        # Make axis equal
        for i in range(num_pars):
            for j in range(num_pars):
                xmin, xmax = axes[i, j].get_xlim()
                xsmin[j].append(xmin)
                xsmax[j].append(xmax)

                if i == j:
                    continue

                ymin, ymax = axes[i, j].get_ylim()
                ysmin[i].append(ymin)
                ysmax[i].append(ymax)

        # Find extrema
        for i in range(num_pars):
            xsmin[i] = min(xsmin[i])
            xsmax[i] = max(xsmax[i])
            ysmin[i] = min(ysmin[i])
            ysmax[i] = max(ysmax[i])

        for i in range(num_pars):
            for j in range(num_pars):
                axes[i, j].set_xlim(xsmin[j], xsmax[j])
                if i == j:
                    continue
                axes[i, j].set_ylim(ysmin[i], ysmax[i])

    ys = np.zeros((asamples.shape[0], problem.y_dim))
    for i, s in enumerate(samples):
        ys[i,:] = problem.statistics(problem.simulator(s))

    plt.figure(1)
    for i in range(problem.y_dim):
        plt.subplot(problem.y_dim, 1, i)
        plt.hist(ys[:,i], 30)
        plt.axvline(problem.y_star[i], c='r')

def plot_statistics(problem, samples):

    asamples = np.array(samples, ndmin=2)

    ys = np.zeros((asamples.shape[0], problem.y_dim))
    for i, s in enumerate(samples):
        ys[i,:] = problem.statistics(problem.simulator(s))

    plt.figure(1)
    for i in range(problem.y_dim):
        plt.subplot(problem.y_dim, 1, i)
        plt.hist(ys[:,i], 30)
        plt.axvline(problem.y_star[i], c='r')


def plot_krs(xs, ts, h, rng, y_star):
    '''
    Plots the kernel surrogate for the given xs and ts.
    `rng` is the range that will be plotted.

    Parameters
    ----------
    xs : np.array
        The x-coordinates of the data points
    ts : np.array
        The corresponding target values
    h : float
        The bandwidth for the kernel regression
    rng : iterable
        List or array of x-coords to plot
    y_star : float
        The target value of the ABC algorithm.

    Note: Works only for 1D problems.
    '''

    means = np.zeros(len(rng))
    stds = np.zeros(len(rng))
    Ns = np.zeros(len(rng))
    confs = np.zeros(len(rng))
    for i, val in enumerate(rng):
        means[i], stds[i], confs[i], Ns[i], _ = \
            km.kernel_regression(val, xs, ts, h)

    plt.fill_between(
        rng,
        means - stds / np.sqrt(Ns),
        means + stds / np.sqrt(Ns),
        color=[0.7, 0.3, 0.3, 0.5])
    plt.fill_between(
        rng,
        means - 2 * stds,
        means + 2 * stds,
        color=[0.7, 0.7, 0.7, 0.7])
    plt.plot(rng, means)
    plt.scatter(xs, ts)
    plt.axhline(y_star)
    plt.ylim(-4, 4)
    #plt.ylim(-4, 14)
    plt.title('S = {0}, h = {1}'.format(len(xs), h))

def peakdet(v, delta, x = None):
    import sys
    from numpy import NaN, Inf, arange, isscalar, array
    '''
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    % [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    % maxima and minima ("peaks") in the vector V.
    % MAXTAB and MINTAB consists of two columns. Column 1
    % contains indices in V, and column 2 the found values.
    %
    % With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    % in MAXTAB and MINTAB are replaced with the corresponding
    % X-values.
    %
    % A point is considered a maximum peak if it has the maximal
    % value, and was preceded (to the left) by a value lower by
    % DELTA.
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    '''
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    #v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


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
