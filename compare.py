'''
Implements currently one distance measure.
'''

import numpy as np

def variation_distance(samples, problem, num_bins=100):
    '''
    Computes the distance of the samples to the true distribution.


    Arguments
    ---------
    samples: array-like
             The list or array of samples
    problem: instance of ABC_Problem
             The problem class that the samples are of. Is used to
             retrieve the cdf of the true posterior.
    num_bins: int
              The number of bins to use for the approximation.

    Returns
    -------
    The difference in histogram heights.
    '''
    diff = np.zeros(num_samples)

    cdf = problem.true_posterior.cdf
    args = problem.true_posterior_args

    for i in xrange(num_samples):
        bins, edges = np.histogram(samples[:i+1], num_bins, normed=True)

        w = float(edges[1] - edges[0])
        edges = np.array(edges)
        bins2 = (cdf(edges[1:], *args) - cdf(edges[:-1], *args)) / w

        vals = np.isfinite(bins2)

        missed_mass = 1.0 - w * sum(bins2[vals])
        diff[i] = 0.5 * (sum(abs(bins[vals] - bins2[vals])) * w + missed_mass)

    return diff


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    from SL_ABC import SL_ABC
    from ASL_ABC import ASL_ABC
    from marginal_ABC import marginal_ABC
    from reject_ABC import reject_ABC
    from problems import toy_problem

    problem = toy_problem()

    num_samples = 3000

    ax1 = plt.subplot(121)
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Error')
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.set_xlabel('Number of simulation calls')

    samples, sim_calls, _ = ASL_ABC(problem, num_samples, 0, 0.05, 10, 5, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='ASL')
    ax2.plot(np.cumsum(sim_calls), dist, label='ASL')

    samples, sim_calls, _ = SL_ABC(problem, num_samples, 0, 10, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='SL')
    ax2.plot(np.cumsum(sim_calls), dist, label='SL')

    samples, sim_calls, _ = marginal_ABC(problem, num_samples, 0.05, 10, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='marginal')
    ax2.plot(np.cumsum(sim_calls), dist, label='marginal')
    
    samples, sim_calls = reject_ABC(problem, num_samples, 0.1, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='rejection')
    ax2.plot(np.cumsum(sim_calls), dist, label='rejection')

    ax1.legend()
    ax2.legend()

    plt.show()
