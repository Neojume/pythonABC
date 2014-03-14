'''
Implements currently one distance measure.
'''

import numpy as np
import data_manipulation as dm
import matplotlib.pyplot as plt
import pickle
import os
import os.path


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

    rng = problem.true_posterior_rng
    edges = np.linspace(rng[0], rng[1], num_bins + 1)
    w = float(edges[1] - edges[0])
    cdf_vals = problem.true_posterior.cdf(edges, *problem.true_posterior_args)
    true_bins = (cdf_vals[1:] - cdf_vals[:-1]) / w

    approx_bins = np.zeros(num_bins)
    missed = 0
    denom = 0

    for i, sample in enumerate(samples):
        if sample < rng[0] or sample > rng[1]:
            missed += 1
        else:
            for b, edge in enumerate(edges):
                if sample < edge:
                    approx_bins[b - 1] += 1
                    break

        denom += w
        normed_bins = approx_bins / denom
        diff[i] = 0.5 * w * (sum(abs(normed_bins - true_bins)) + missed / denom)

    return diff


def plot_distances(problem, num_samples, methods, method_args, method_labels,
                   repeats=1, verbose=False):
    '''
    Retrieves results from data files and plots the results.

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
    repeats : int
        Number of times runs are repeated.
    verbose : bool
        If True prints iteration numbers
    '''

    ax1 = plt.subplot(121)
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Error')
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.set_xlabel('Number of simulation calls')

    for i, method in enumerate(methods):
        # Allocate memory
        dist = np.zeros((num_samples, repeats))
        sim_calls = np.zeros((num_samples, repeats))

        filename = dm.get_filename(method, method_args[i], problem)
        path = os.path.join(os.getcwd(), 'data', filename)
        with open(path, 'rb') as f:
            alg_data = pickle.load(f)

        data_index = 0
        array_index = 0

        while array_index < repeats:
            print 'data entry', data_index
            if data_index > len(alg_data.list_of_samples):
                raise Exception('Not enough runs of algorithm: {0}'.format(
                                method.__name__))

            samples = alg_data.list_of_samples[data_index]
            if len(samples) < num_samples:
                data_index += 1
                continue

            sim_calls[:, array_index] = alg_data.list_of_sim_calls[data_index]
            dist[:, array_index] = variation_distance(samples, problem)

            array_index += 1
            data_index += 1

        avg_dist = np.mean(dist, 1)
        std_dist = np.std(dist, 1)
        avg_sim_calls = np.mean(np.cumsum(sim_calls, 0), 1)

        line, = ax1.plot(avg_dist, label=method_labels[i])
        ax1.fill_between(
            range(num_samples),
            avg_dist - 2 * std_dist,
            avg_dist + 2 * std_dist,
            color=line.get_color(),
            alpha=0.5)

        # TODO: Compute errors on the sim_call average
        line, = ax2.plot(avg_sim_calls, avg_dist, label=method_labels[i])
        ax2.fill_between(
            avg_sim_calls,
            avg_dist - 2 * std_dist,
            avg_dist + 2 * std_dist,
            color=line.get_color(),
            alpha=0.5)

    ax1.legend()
    ax2.legend()
    ax1.set_xscale('log')
    ax2.set_xscale('log')


def call_and_plot_distances(
    problem, num_samples, methods, method_args, method_labels,
        repeats=1, verbose=False):
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
    repeats : int
        Number of times runs are repeated.
    verbose : bool
        If True prints iteration numbers
    '''

    ax1 = plt.subplot(121)
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Error')
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.set_xlabel('Number of simulation calls')

    for i, method in enumerate(methods):
        # Allocate memory
        dist = np.zeros((num_samples, repeats))
        sim_calls = np.zeros((num_samples, repeats))

        for j in range(repeats):
            if verbose:
                print 'Algorithm label:', method_labels[i], 'repetition', j
            return_tuple = method(problem, num_samples, *method_args[i])

            samples = return_tuple[0]
            sim_calls[:, j] = return_tuple[1]
            dist[:, j] = variation_distance(samples, problem)

        avg_dist = np.mean(dist, 1)
        std_dist = np.std(dist, 1)
        avg_sim_calls = np.mean(np.cumsum(sim_calls, 0), 1)

        line, = ax1.plot(avg_dist, label=method_labels[i])
        ax1.fill_between(
            range(num_samples),
            avg_dist - 2 * std_dist,
            avg_dist + 2 * std_dist,
            color=line.get_color(),
            alpha=0.5)

        # TODO: Compute errors on the sim_call average
        line, = ax2.plot(avg_sim_calls, avg_dist, label=method_labels[i])
        ax2.fill_between(
            avg_sim_calls,
            avg_dist - 2 * std_dist,
            avg_dist + 2 * std_dist,
            color=line.get_color(),
            alpha=0.5)

    ax1.legend()
    ax2.legend()


if __name__ == '__main__':
    from SL_ABC import SL_ABC
    from ASL_ABC import ASL_ABC
    from KRS_ABC import KRS_ABC
    from marginal_ABC import marginal_ABC, pseudo_marginal_ABC
    from reject_ABC import reject_ABC
    from problems import *

    problem = wilkinson_problem()

    num_samples = 40000
    repeats = 2

    methods = []
    method_args = []
    method_labels = []

    # methods.append(ASL_ABC)
    # method_labels.append('ASL_ABC')
    #method_args.append([0.05, 0.1, 5, 5])

    # methods.append(SL_ABC)
    #method_labels.append('SL_ABC 2')
    #method_args.append([0.05, 2, True])

    methods.append(marginal_ABC)
    method_labels.append('marginal ABC')
    method_args.append([0.05, 500, True])

    methods.append(pseudo_marginal_ABC)
    method_labels.append('pseudo marginal ABC')
    method_args.append([0.05, 1000, True])

    # methods.append(reject_ABC)
    #method_labels.append('reject_ABC 0.1')
    #method_args.append([0.1, True])

    call_and_plot_distances(problem,
                            num_samples,
                            methods,
                            method_args,
                            method_labels,
                            repeats)
    plt.show()
