'''
An example of how to implement a new problem class, perform some experiments on
it and use some other functions of this lib.

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
import pythonABC.data_manipulation as dm


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
#   The used algorithm is the Synthetic Likelihood by Wood.
#   The epsilon tube for the algorithm is set to 0.05.
#   The number of simulations to perform each iteration is set to 10.
#   The covariance matrix is forced to be diagonal.
#   The algorithm will print iteration number and number of simulation  calls
#   to the console.
#   There is no data directory to store the results in.
#   The results will not be saved.
# Note that by default the result will be stored in a .abc file. This behaviour
# can be turned of by setting the option `save` to False.
experiment = algs.SL_ABC(
        problem_instance,
        epsilon=0.05,
        S=10,
        diag=True,
        verbose=True,
        data_dir=False,
        save=False)

# Run the experiment for 5000 samples
experiment.run(5000)

# Plot the samples using a histogram. The plot function in utils will also plot
# the true posterior density if set in the problem class.
utils.plot_samples(problem_instance, experiment.samples)

# Show the results
plt.show()

# The experiment can be run again with another call to the run method. If we
# wanted to continue from the point where the Markov chain stopped after the
# last run terminated, the run method can be invoked with the option `reset`
# set to False.

# Continue the experiment for an additional 5000 samples.
experiment.run(5000, reset=False)

# Here we should have 10000 samples in our experiment
assert len(experiment.samples) == 10000

# Because the save flag was set to False, save manually.
# Filename is generated using the name of the algorithm, the settings of its
# parameters and the problem that its solving.
dm.save(experiment)

# The load function can be used to load previous experiments data.
# Note that the experiment object is used to determine which .abc file to
# load. The returned data object can contain data for more than 1 experiment.
data = dm.load(experiment)

# The data object contains different fields for the information that is stored.
# There is for example a list containing lists of samples. The first entry of
# this list is the list of samples of our experiment. (Given that we did not run
# any other experiments before this one).
assert data.list_of_samples[0] == experiment.samples

# There are a couple of other fields, see the ABCData class in the data
# manipulation file.
