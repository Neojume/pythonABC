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

    # Calculate true bin values
    rng = problem.true_posterior_rng
    edges = np.linspace(rng[0], rng[1], num_bins + 1)
    w = float(edges[1] - edges[0])
    cdf_vals = problem.true_posterior.cdf(edges, *problem.true_posterior_args)
    true_bins = (cdf_vals[1:] - cdf_vals[:-1]) / w

    # Allocate memory for approximate bins
    approx_bins = np.zeros(num_bins)
    missed = 0
    denom = 0

    for i, sample in enumerate(samples):
        # Put this sample in the correct bin (or missed)
        if sample < rng[0] or sample > rng[1]:
            missed += 1
        else:
            b = np.searchsorted(edges, sample)
            approx_bins[b - 1] += 1

        # Calculate the difference between the bins
        denom += w
        normed_bins = approx_bins / denom
        diff[i] = 0.5 * w * \
            (sum(abs(normed_bins - true_bins)) + missed / denom)

    return diff


def plot_distances(problem, num_samples, methods, method_args, method_labels,
                   repeats=1, call=True, verbose=False):
    '''
    Retrieves results from data files and plots the results. If there are not
    enough data points for either the number of repeats or number of samples,
    additional data points are added (functions are called)

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
    call : bool
       If True will do additional runs of the algorithms to obtain enough data.
       Default True.
    verbose : bool
        If True prints iteration numbers, default False.
    '''

    ax1 = plt.subplot(121)
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Error')
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.set_xlabel('Number of simulation calls')

    for i, method in enumerate(methods):
        dist = np.zeros((num_samples, repeats))
        sim_calls = np.zeros((num_samples, repeats))

        filename = dm.get_filename(method, method_args[i], problem)
        path = os.path.join(os.getcwd(), 'data', filename)
        try:
            # Try to read from the database
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)
        except IOError:
            # The file doesn't exist: run the experiments
            for r in range(repeats):
                method(problem, num_samples, *method_args[i])

            # And load the data after
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)

        data_index = 0
        array_index = 0

        while array_index < repeats:
            print 'data entry', data_index
            if data_index > len(alg_data.list_of_samples):
                if not call:
                    raise Exception('Not enough data points')
                # Run the required additional experiments
                for r in range(repeats - array_index):
                    method(problem, num_samples, *method_args[i])

                # Reload the algorithm data
                with open(path, 'rb') as f:
                    alg_data = pickle.load(f)

                continue

            samples = alg_data.list_of_samples[data_index]
            if len(samples) < num_samples:
                data_index += 1
                continue

            samples = samples[:num_samples]

            sim_calls[:, array_index] = alg_data.list_of_sim_calls[
                data_index][:num_samples]
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
