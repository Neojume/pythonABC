import distributions as distr
import numpy as np


class ABC_Problem(object):
    y_star = None
    y_dim = None

    prior = None
    prior_args = None

    proposal = None
    proposal_args = None

    theta_init = None

    def simulator(self, theta):
        raise NotImplemented

    def real_posterior(self, x):
        raise NotImplemented


class toy_problem(ABC_Problem):
    '''
    The exponential toy problem of Meeds and Welling.
    '''

    def __init__(self, y_star=9.42, N=500):
        self.y_star = y_star
        self.y_dim = 1

        self.N = N

        self.prior = distr.gamma
        self.prior_args = [0.1, 0.1]

        self.proposal = distr.lognormal
        self.proposal_args = [0.1]

        self.theta_init = 0.1

        self.true_posterior = distr.gamma
        self.true_posterior_args = [
            self.prior_args[0] + self.N,
            self.prior_args[1] + self.N * self.y_star]

    def simulator(self, theta):
        return np.mean(distr.exponential.rvs(theta, self.N))

    def true_function(self, theta):
        return 1.0 / theta


class sinus_problem(ABC_Problem):
    def __init__(self):
        self.y_star = 1.3
        self.y_dim = 1

        self.prior = distr.uniform
        self.prior_args = [0, 4 * np.pi]

        self.proposal = distr.normal
        self.proposal_args = [3]

        self.theta_init = 0.5

    def true_function(self, theta):
        return np.sin(theta / 1.2) + 0.1 * theta

    def simulator(self, theta):
        return np.sin(theta / 1.2) + 0.1 * theta + distr.normal.rvs(0, 0.2)

    def real_posterior(x):
        raise NotImplemented
