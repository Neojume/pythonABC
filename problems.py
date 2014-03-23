import distributions as distr
import numpy as np
from scipy.integrate import quad
import collections


class ABC_Problem(object):
    # The dimensionality and value of y_star
    y_star = None
    y_dim = None

    # Prior distribution class and its arguments
    prior = None
    prior_args = None

    # The range for the problem (used for visualisation)
    rng = None

    # Proposal distribution class and its arguments
    proposal = None
    proposal_args = None

    # Whether the proposal distribution was specified in terms of
    # log(theta)
    use_log = None

    # The initial value of theta
    theta_init = None

    # The true posterior distribution class and its arguments.
    # Optional:
    # Note that this is used for comparison of convergence.
    true_posterior_rng = None
    true_posterior = None
    true_posterior_args = None

    def simulator(self, theta):
        raise NotImplemented

    def real_posterior(self, x):
        raise NotImplemented


class exponential_problem(ABC_Problem):
    '''
    The exponential toy problem of Meeds and Welling.
    '''

    def __init__(self, y_star=9.42, N=500):
        self.y_star = y_star
        self.y_dim = 1

        self.N = N

        self.prior = distr.gamma
        self.prior_args = [0.1, 0.1]

        self.rng = [1e-10, 10]

        self.proposal = distr.lognormal
        self.proposal_args = [0.1]
        self.use_log = True

        self.theta_init = 0.1

        self.true_posterior_rng = [0.05, 0.15]
        self.true_posterior = distr.gamma
        self.true_posterior_args = [
            self.prior_args[0] + self.N,
            self.prior_args[1] + self.N * self.y_star]

    def simulator(self, theta):
        return np.mean(distr.exponential.rvs(theta, self.N))

    def true_function(self, theta):
        return 1.0 / theta


class wilkinson_problem(ABC_Problem):
    '''
    Toy problem of Richard Wilkinson from his NIPS tutorial.
    '''

    def __init__(self):
        self.y_star = 2.0
        self.y_dim = 1

        self.prior = distr.uniform
        self.prior_args = [-10, 10]

        self.true_posterior_rng = [-3.5, 3.5]
        proportional = lambda x: distr.normal.pdf(
            self.y_star, self.true_function(x), 0.1 + x ** 2)
        self.true_posterior = distr.proportional(proportional)
        self.true_posterior_args = []

        self.rng = [-10, 10]

        self.proposal = distr.normal
        self.proposal_args = [2]
        self.use_log = False

        self.theta_init = 0.5

    def true_function(self, theta):
        return 2 * (theta + 2) * theta * (theta - 2)

    def simulator(self, theta):
        mu = 2 * (theta + 2) * theta * (theta - 2)
        return distr.normal.rvs(mu, 0.1 + theta ** 2)[0]


class sinus_problem(ABC_Problem):
    '''
    Sinus problem.
    '''

    def __init__(self):
        self.y_star = 1.3
        self.y_dim = 1

        self.prior = distr.uniform
        self.prior_args = [0, 4 * np.pi]

        self.rng = [0, 4 * np.pi]

        self.proposal = distr.normal
        self.proposal_args = [3]
        self.use_log = False

        self.true_posterior_rng = self.rng
        proportional = lambda x: distr.normal.pdf(
            self.y_star, self.true_function(x), 0.2)
        self.true_posterior = distr.proportional(proportional)
        self.true_posterior_args = []

        self.theta_init = 0.5

    def true_function(self, theta):
        return np.sin(theta / 1.2) + 0.1 * theta

    def simulator(self, theta):
        return np.sin(theta / 1.2) + 0.1 * theta + distr.normal.rvs(0, 0.2)[0]
