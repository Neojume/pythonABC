import distributions as distr
import numpy as np
from scipy.integrate import quad
import collections

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
        self.use_log = True

        self.theta_init = 0.1

        self.true_posterior = distr.gamma
        self.true_posterior_args = [
            self.prior_args[0] + self.N,
            self.prior_args[1] + self.N * self.y_star]

    def simulator(self, theta):
        return np.mean(distr.exponential.rvs(theta, self.N))

    def true_function(self, theta):
        return 1.0 / theta


class wilkinson_problem(ABC_Problem):

    class wilkinson_posterior(object):
        def __init__(self, y_star, true_function):
            # Calculate normalization beforehand
            self.y_star = y_star
            self.true_function = true_function
            self.normalization, _ = quad(
                self.proportional_posterior,
                -np.inf,
                np.inf)

            self.rng = np.linspace(-20, 20, 1000)
            self.cdf_list = [quad(self.pdf, -np.inf, i)[0] for i in self.rng]

        def proportional_posterior(self, x):
            return distr.normal.pdf(
                self.y_star,
                self.true_function(x),
                0.1 + x ** 2)

        def pdf(self, x):
            return self.proportional_posterior(x) / self.normalization

        def cdf(self, x):
            return np.interp(x, self.rng, self.cdf_list)

            #if isinstance(x, collections.Iterable):
            #    val = np.array([quad(self.pdf, -np.inf, i)[0] for i in x])
            #else:
            #    val = quad(self.pdf, -np.inf, x)[0]
            #return val

    def __init__(self):
        self.y_star = 2.0
        self.y_dim = 1

        self.prior = distr.uniform
        self.prior_args = [-10, 10]

        self.true_posterior = wilkinson_problem.wilkinson_posterior(
            self.y_star, self.true_function)
        self.true_posterior_args = []

        self.proposal = distr.normal
        self.proposal_args = [1]
        self.use_log = False

        self.theta_init = 0.5

    def true_function(self, theta):
        return 2 * (theta + 2) * theta * (theta - 2)

    def simulator(self, theta):
        mu = 2 * (theta + 2) * theta * (theta - 2)
        return distr.normal.rvs(mu, 0.1 + theta ** 2)[0]

class sinus_problem(ABC_Problem):
    def __init__(self):
        self.y_star = 1.3
        self.y_dim = 1

        self.prior = distr.uniform
        self.prior_args = [0, 4 * np.pi]

        self.proposal = distr.normal
        self.proposal_args = [3]
        self.use_log = False

        self.theta_init = 0.5

    def true_function(self, theta):
        return np.sin(theta / 1.2) + 0.1 * theta

    def simulator(self, theta):
        return np.sin(theta / 1.2) + 0.1 * theta + distr.normal.rvs(0, 0.2)[0]

    def real_posterior(x):
        raise NotImplemented
