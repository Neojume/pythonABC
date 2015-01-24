'''
An example of how to implement a new problem class and perform
some experiments on it.

@author Steven
'''

import numpy as np
import matplotlib.pyplot as plt

import sys

import pythonABC
from pythonABC.problems import ABC_Problem

# Some shorthands
import pythonABC.distributions as distr
import pythonABC.algorithms as algs
import pythonABC.utils as utils
import pythonABC.kernels as kernels


class Exponential_Problem(ABC_Problem):

    '''
    The exponential toy problem of Meeds and Welling.
    '''

    def __init__(self, y_star=9.42, true_args=0.1, N=500):

        # The dimensionality and value of y_star
        self.y_star = np.array([y_star])
        self.y_dim = 1

        self.N = N

        # Prior distribution class and its arguments
        self.prior = distr.gamma
        self.prior_args = [0.1, 0.1]

        # The range for the problem (used for visualisation)
        self.rng = [1e-10, 10]

        # Proposal distribution class and its arguments
        self.proposal = distr.lognormal
        self.proposal_args = [0.1]
        self.use_log = True

        # The true posterior distribution class and its arguments.
        self.true_posterior = distr.gamma
        self.true_posterior_args = [
            self.prior_args[0] + self.N,
            self.prior_args[1] + self.N * self.y_star]

        # The range to be plotted of the posterior
        self.true_posterior_rng = [0.05, 0.15]

        # List of labels of simulator args (what each dimension of theta is)
        self.simulator_args = ['theta']

        # The arguments used to obtain the y_star
        self.true_args = np.array([true_args])

    def get_theta_init(self):
        return self.prior.rvs(*self.prior_args)

    def statistics(self, vals):
        return np.mean(vals, keepdims=True)

    def simulator(self, theta):
        return distr.exponential.rvs(theta, self.N)

    def true_function(self, theta):
        return 1.0 / theta

# Create an instance of the problem
problem_instance = Exponential_Problem()

# Create the experiment where:
#   The epsilon tube for the algorithm is set to 0.05.
#   The number of simulations to perform each iteration is set to 10.
#   The covariance matrix is forced to be diagonal.
#   The algorithm will print iteration number and number of simulation
#   calls to the console.
#   There is no data directory to store the results in.
experiment = algs.SL_ABC(
        problem_instance,
        epsilon=0.05,
        S=10,
        diag=True,
        verbose=True,
        data_dir=False)

# Run the experiment for 5000 samples
experiment.run(5000)

# Plot the last 5000 samples using a histogram
utils.plot_samples(problem_instance, experiment.samples)

# Show the results
plt.show()
