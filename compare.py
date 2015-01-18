'''
Implements currently one distance measure.
'''

import numpy as np
import data_manipulation as dm
import matplotlib.pyplot as plt
import pickle
import os
import os.path


__all__ = ['NMSE_convergence',
           'variation_distance',
           'plot_distances',
           'plot_convergence',
           'run_algorithms']


def NMSE_convergence(problem, samples):
    cur_avg = np.zeros(problem.y_dim)

    NMSE = np.zeros((problem.y_dim, len(samples)))
    sims = np.zeros((problem.y_dim, len(samples)))
    y_star = problem.y_star
    numer = np.zeros(problem.y_dim)

    for N, sample in enumerate(samples):
        sims[:, N] = problem.statistics(problem.simulator(sample))
        cur_avg += (sims[:, N] - cur_avg) / float(N + 1)
        NMSE[:, N] = pow(cur_avg - y_star, 2) / pow(y_star, 2)

    return NMSE


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


def plot_distances(problem, num_samples, algorithms, algorithm_labels,
                   repeats=1, call=True, verbose=False, which='Both'):
    '''
    Retrieves results from data files and plots the results. If there are not
    enough data points for either the number of repeats or number of samples,
    additional data points are added (functions are called)

    The left plot is the number of samples against the error. The right plot is
    the number of simulation calls against the error.

    Arguments
    ---------
    problem : instance of ABC_Problem
        The problem we're trying to solve.
    num_samples : int
        The number of samples to draw.
    algorithms : list
        List of algorithms to plot the curves of.
    algorithm_labels : list of strings
        Labels for the legend of the plot.

    Optional Arguments
    ------------------
    repeats : int
        Number of times runs are repeated.
    call : bool
       If True will do additional runs of the algorithms to obtain enough data.
       Default True.
    which : string
        Which of the plots to show.
        `Samples`
        `Simulations`
        `Both`
    verbose : bool
        If True prints iteration numbers, default False.

    Returns
    -------
    ax1, ax2 : tuple
        The axis handles of the two created figures
    '''

    color_cycle = ['r', '0', 'b', 'g']

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_color_cycle(color_cycle)
    ax1.set_xlabel(r'Number of samples')
    ax1.set_ylabel(r'Total Variation Distance')

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.set_color_cycle(color_cycle)
    ax2.set_xlabel(r'Number of simulation calls')
    ax2.set_ylabel(r'Total Variation Distance')

    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.set_color_cycle(color_cycle)
    ax3.set_xlabel(r'Number of samples')
    ax3.set_ylabel(r'Number of simulation calls')

    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(111)
    ax4.set_color_cycle(color_cycle)
    ax4.set_xlabel(r'Number of simulation calls')
    ax4.set_ylabel(r'NMSE')

    for i, algorithm in enumerate(algorithms):
        dist = np.zeros((num_samples, repeats))
        sim_calls = np.zeros((num_samples, repeats))
        nmse = np.zeros((num_samples, repeats))

        filename = dm.get_filename(algorithm)
        path = os.path.join(os.getcwd(), 'data', filename)
        try:
            # Try to read from the database
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)
        except IOError:
            # The file doesn't exist: run the experiments
            for r in range(repeats):
                algorithm.run(num_samples)

            # And load the data after
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)

        data_index = 0
        array_index = 0

        while array_index < repeats:
            print 'data entry', data_index
            if data_index >= len(alg_data.list_of_samples):
                if not call:
                    raise Exception('Not enough data points')
                # Run the required additional experiments
                for r in range(repeats - array_index):
                    algorithm.run(num_samples)

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
            nmse[:, array_index] = NMSE_convergence(problem, samples)

            array_index += 1
            data_index += 1

        avg_dist = np.mean(dist, 1)
        std_dist = np.std(dist, 1)
        avg_sim_calls = np.mean(np.cumsum(sim_calls, 0), 1)
        std_sim_calls = np.std(np.cumsum(sim_calls, 0), 1)
        avg_nmse = np.mean(nmse, 1)
        std_nmse = np.std(nmse, 1)

        indices = np.floor(np.logspace(0, np.log10(num_samples))).astype(int)
        indices[0] = 0
        indices[-1] = num_samples - 1

        line, = ax1.plot(indices, avg_dist[indices], label=algorithm_labels[i])
        ax1.fill_between(
            indices,
            avg_dist[indices] - 2 * std_dist[indices],
            avg_dist[indices] + 2 * std_dist[indices],
            color=line.get_color(),
            alpha=0.25)

        line, = ax2.plot(avg_sim_calls[indices], avg_dist[indices], label=algorithm_labels[i])
        ax2.fill_between(
            avg_sim_calls[indices],
            avg_dist[indices] - 2 * std_dist[indices],
            avg_dist[indices] + 2 * std_dist[indices],
            color=line.get_color(),
            alpha=0.25)

        line, = ax3.plot(range(num_samples), avg_sim_calls, label=algorithm_labels[i])
        ax3.fill_between(
            np.array(range(num_samples), ndmin=1),
            np.clip(avg_sim_calls - 2 * std_sim_calls, 1, np.inf),
            avg_sim_calls + 2 * std_sim_calls,
            color=line.get_color(),
            alpha=0.25)

        line, = ax4.plot(avg_sim_calls, avg_nmse, label=algorithm_labels[i])
        ax4.fill_between(
            avg_sim_calls,
            avg_nmse - 2 * std_nmse,
            avg_nmse + 2 * std_nmse,
            color=line.get_color(),
            alpha=0.25)

    ax1.set_xscale('log')
    ax1.grid(True)
    ax1.legend(loc='best')

    ax2.set_xscale('log')
    ax2.grid(True)
    ax2.legend(loc='best')

    ax3.set_xscale('log')
    ax3.set_yscale('log', nonposy='clip')
    ax3.grid(True)
    ax3.legend(loc='best')

    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True)
    ax4.legend(loc='best')

    return ax1, ax2, ax3, ax4

def run_algorithms(problem, num_samples, algorithms, algorithm_labels,
                   repeats=1, call=True, verbose=False):
    '''
    '''

    for i, algorithm in enumerate(algorithms):
        filename = dm.get_filename(algorithm)
        path = os.path.join(os.getcwd(), 'data', filename)
        try:
            # Try to read from the database
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)
        except IOError:
            # The file doesn't exist: run the experiments
            for r in range(repeats):
                algorithm.run(num_samples)

            # And load the data after
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)

        data_index = 0
        array_index = 0

        while array_index < repeats:
            print 'data entry', data_index
            if data_index >= len(alg_data.list_of_samples):
                if not call:
                    raise Exception('Not enough data points')
                # Run the required additional experiments
                for r in range(repeats - array_index):
                    algorithm.run(num_samples)

                # Reload the algorithm data
                with open(path, 'rb') as f:
                    alg_data = pickle.load(f)

                continue

            samples = alg_data.list_of_samples[data_index]
            if len(samples) < num_samples:
                data_index += 1
                continue

            array_index += 1
            data_index += 1

def plot_convergence(problem, num_samples, algorithms, algorithm_labels,
                   repeats=1, call=True, verbose=False):
    '''
    Retrieves results from data files and plots the results. If there are not
    enough data points for either the number of repeats or number of samples,
    additional data points are added (functions are called)

    The left plot is the number of samples against the error. The right plot is
    the number of simulation calls against the error.

    Arguments
    ---------
    problem : instance of ABC_Problem
        The problem we're trying to solve.
    num_samples : int
        The number of samples to draw.
    algorithms : list
        List of algorithms to plot the curves of.
    algorithm_labels : list of strings
        Labels for the legend of the plot.

    Optional Arguments
    ------------------
    repeats : int
        Number of times runs are repeated.
    call : bool
       If True will do additional runs of the algorithms to obtain enough data.
       Default True.
    which : string
        Which of the plots to show.
        `Samples`
        `Simulations`
        `Both`
    verbose : bool
        If True prints iteration numbers, default False.

    Returns
    -------
    ax1, ax2 : tuple
        The axis handles of the two created figures
    '''

    color_cycle = ['r', 'b', '0', 'g']

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.set_color_cycle(color_cycle)
    ax1.set_xlabel(r'Number of simulation calls')
    ax1.set_ylabel(r'NMSE')

    for i, algorithm in enumerate(algorithms):
        sim_calls = np.zeros((num_samples, repeats))
        nmse = np.zeros((num_samples, repeats))

        filename = dm.get_filename(algorithm)
        path = os.path.join(os.getcwd(), 'data', filename)
        try:
            # Try to read from the database
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)
        except IOError:
            # The file doesn't exist: run the experiments
            for r in range(repeats):
                algorithm.run(num_samples)

            # And load the data after
            with open(path, 'rb') as f:
                alg_data = pickle.load(f)

        data_index = 0
        array_index = 0

        while array_index < repeats:
            print 'data entry', data_index
            if data_index >= len(alg_data.list_of_samples):
                if not call:
                    raise Exception('Not enough data points')
                # Run the required additional experiments
                for r in range(repeats - array_index):
                    algorithm.run(num_samples)

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
            nmse[:, array_index] = NMSE_convergence(problem, samples)

            array_index += 1
            data_index += 1

        avg_sim_calls = np.mean(np.cumsum(sim_calls, 0), 1)
        std_sim_calls = np.std(np.cumsum(sim_calls, 0), 1)
        avg_nmse = np.mean(nmse, 1)
        std_nmse = np.std(nmse, 1)

        line, = ax1.plot(avg_sim_calls, avg_nmse, label=algorithm_labels[i])
        ax1.fill_between(
            avg_sim_calls,
            avg_nmse - 2 * std_nmse,
            avg_nmse + 2 * std_nmse,
            color=line.get_color(),
            alpha=0.25)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()

    return ax1
