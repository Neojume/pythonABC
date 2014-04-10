import numpy as np
import matplotlib.pyplot as plt
import kernel_methods as km


def get_weighted_bootstrap(values, weights, num_samples):
    '''
    Get a bootstrap sample of the given values, each value picked with the
    given weight.
    '''

    values = np.asarray(values)
    c = np.cumsum(weights)
    r = np.random.uniform(0, sum(weights), num_samples)
    return values[np.searchsorted(c, r)]


def get_bootstrap(samples):
    '''
    Get a bootstrap sample of the given samples.
    '''

    samples = np.asarray(samples)
    ids = np.random.random_integers(0, len(samples) - 1, len(samples))

    return samples[ids]


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

    asamples = np.array(samples)
    print asamples.shape
    pars = problem.simulator_args
    num_pars = len(pars)
    has_true_vals = problem.true_args is not None

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

    for i, par1 in enumerate(pars):
        for j, par2 in enumerate(pars):
            axes[i, j] = plt.subplot(num_pars, num_pars, num_pars * i + j + 1)

    for i, par1 in enumerate(pars):
        for j, par2 in enumerate(pars):
            ax = axes[i, j]

            if i == j:
                # Same parameter, so make a histogram
                ax.hist(par_values[par1], 50)
                if has_true_vals:
                    ax.axvline(problem.true_args[i], color='r')
            else:
                # Different parameters, make scatterplot
                ax.scatter(par_values[par2], par_values[par1], alpha=0.1)
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
