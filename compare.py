'''
Implements currently one distance measure.
'''

import numpy as np

def variation_distance(samples, problem, num_bins=100):
    '''
    Computes the distance of the samples to the true distribution.

    Arguments
    ---------
    samples : array-like
        The list or array of samples
    problem : instance of ABC_Problem
        The problem class that the samples are of. Is used to retrieve the cdf 
        of the true posterior.
    num_bins: int
        The number of bins to use for the approximation.

    Returns
    -------
    distance : array
        The difference in histogram heights for each number of samples.
    '''

    num_samples = len(samples)
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

def plot_distances(problem, num_samples, methods, method_args, method_labels, 
        repeats=1):
    '''
    Calls the methods and plots the results.

    Arguments
    ---------
    problem : instance of ABC_Problem
        The problem we're trying to solve.
    num_samples : int
        The number of samples to draw.
    methods : list
        List of methods to plot the curves of.
    method_args : list of lists
        Additional arguments for the different methods. problem and number of
        samples will always be passed to the methods.
    method_labels : list of strings
        Labels for the legend of the plot.
    '''
    ax1 = plt.subplot(121)
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Error')
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.set_xlabel('Number of simulation calls')

    for i, method in enumerate(methods):
        dist = np.zeros((num_samples, repeats))
        sim_calls = np.zeros((num_samples, repeats))
        for j in range(repeats):
            return_tuple = method(problem, num_samples, *method_args[i])

            samples = return_tuple[0]
            sim_calls[:, j] = return_tuple[1]
            dist[:,j] = variation_distance(samples, problem)

        avg_dist = np.mean(dist, 1)
        avg_sim_calls = np.mean(np.cumsum(sim_calls, 0), 1)

        ax1.plot(avg_dist, label=method_labels[i])
        ax2.plot(avg_sim_calls, avg_dist, label=method_labels[i])

    ax1.legend()
    ax2.legend()

    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    from SL_ABC import SL_ABC
    from ASL_ABC import ASL_ABC
    from KRS_ABC import KRS_ABC
    from marginal_ABC import marginal_ABC
    from reject_ABC import reject_ABC
    from problems import toy_problem

    problem = toy_problem()

    num_samples = 2000
    repeats = 10

    methods = []
    method_args = []
    method_labels = []

    methods.append(KRS_ABC)
    method_labels.append('KRS_ABC')
    method_args.append([0.0, 0.1, 20, 10, True])
    
    '''
    methods.append(ASL_ABC)
    method_labels.append('ASL_ABC')
    method_args.append([0, 0.05, 10, 5, True])

    methods.append(SL_ABC)
    method_labels.append('SL_ABC')
    method_args.append([0.05, 10, True])

    methods.append(marginal_ABC)
    method_labels.append('marginal_ABC')
    method_args.append([0.05, 10, True])
 
    methods.append(reject_ABC)
    method_labels.append('reject_ABC')
    method_args.append([0.1, True])
    '''

    plot_distances(problem, 
            num_samples, 
            methods, 
            method_args, 
            method_labels, 
            repeats)
