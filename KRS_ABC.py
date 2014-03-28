# -*- coding: utf-8 -*-
"""
KRS-ABC

Kernel Regressian Surrogate - Approximate Bayesian Computation

@author: steven
"""

# Used libraries
import matplotlib.pyplot as plt
import numpy as np

# Own imports
import distributions as distr
from kernel_regression import kernel_regression
import data_manipulation as dm
from algorithms import Base_MCMC_ABC_Algorithm
from utils import conditional_error


def plot_krs(xs, ts, h, rng, y_star):
    means = np.zeros(50)
    stds = np.zeros(50)
    Ns = np.zeros(50)
    confs = np.zeros(50)
    for i, val in enumerate(rng):
        means[i], stds[i], confs[i], Ns[i] = kernel_regression(val, xs, ts, h)

    plt.fill_between(
        rng,
        means - stds / np.sqrt(Ns),
        means + stds / np.sqrt(Ns),
        color=[0.7, 0.3, 0.3, 0.5])
    plt.fill_between(
        rng,
        means - 2 * stds,
        means + 2 * stds,
        color=[0.7, 0.7, 0.7, 0.7])
    plt.plot(rng, problem.true_function(rng))
    plt.plot(rng, means)
    plt.scatter(xs, ts)
    plt.axhline(y_star)
    plt.ylim(-4, 4)
    plt.title('S = {0}, h = {1}'.format(len(xs), h))


class KRS_ABC(Base_MCMC_ABC_Algorithm):
    def __init__(self, problem, num_samples, **params):
        '''
        Creates an instance of the Kernel Regression Surrogate ABC algorithm,
        which is the same as GPS-ABC described by Meeds and Welling [1]_, only
        with a kernel regression instead of a Gaussian process.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        num_samples : int
            The number of samples
        epsilon : float
            Epsilon for the error tube
        ksi : float
            Error margin
        S0 : int
            Number of initial simulations per iteration
        delta_S : int
            Number of additional simulations

        Note that while epsilon, ksi, S0 and delta_S are keyword-arguments,
        they are necessary.

        Optional Arguments
        ------------------
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulations. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database
            Default `True`.
        M : int
            Number of samples to approximate mu_hat. Default 50.
        E : int
            Number of points to approximate conditional error. Default 50.

        References
        ----------
        .. [1] GPS-ABC: Gaussian Process Surrogate Approximate Bayesian
           Computation. E. Meeds and M. Welling.
           http://arxiv.org/abs/1401.2838
        '''
        super(KRS_ABC, self).__init__(problem, num_samples, **params)

        self.needed_params = ['epsilon', 'ksi', 'S0', 'delta_S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S0 = params['S0']
        self.epsilon = params['epsilon']
        self.ksi = params['ksi']
        self.delta_S = params['delta_S']

        if 'M' in params.keys():
            self.M = params['M']
        else:
            self.M = 50

        if 'E' in params.keys():
            self.E = params['E']
        else:
            self.E = 50

        # TODO: Set this more intelligently
        self.h = 0.1

        self.eps_eye = np.identity(self.y_dim) * self.epsilon ** 2

        self.xs = list(self.prior.rvs(*self.prior_args)
                       for s in xrange(self.S0))
        self.ts = [self.simulator(x) for x in self.xs]

    #def mh_step(self):
    #    while True:
    #        # Get mus and sigmas from Kernel regression
    #        xs_a = np.array(self.xs, ndmin=2).T
    #        ts_a = np.array(self.ts, ndmin=2).T

    def mh_step(self):
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        while True:
            # Get mus and sigmas from Kernel regression
            xs_a = np.array(self.xs, ndmin=2).T
            ts_a = np.array(self.ts, ndmin=2).T

            mu_bar, std, conf, N, cov = kernel_regression(
                np.array(self.theta), xs_a, ts_a, self.h)
            mu_bar_p, std_p, conf_p, N_p, cov_p = kernel_regression(
                np.array(self.theta_p), xs_a, ts_a, self.h)

            # Get samples
            mu = distr.normal.rvs(mu_bar, std / np.sqrt(N), self.M)
            mu_p = distr.normal.rvs(mu_bar_p, std_p / np.sqrt(N_p), self.M)

            # TODO: Make multidimensional work.
            alphas = np.zeros(self.M)
            for m in xrange(self.M):
                other_term = \
                    distr.multivariate_normal.logpdf(
                        self.y_star,
                        mu_p[m],
                        cov_p + self.eps_eye) - \
                    distr.multivariate_normal.logpdf(
                        self.y_star,
                        mu[m],
                        cov + self.eps_eye)

                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error > self.ksi:
                # Acquire training points
                for s in xrange(self.delta_S):
                    new_x = self.prior.rvs(*self.prior_args)
                    self.ts.append(self.simulator(new_x))
                    self.xs.append(new_x)
                    self.current_sim_calls += 1
            else:
                break

        return distr.uniform.rvs() <= tau


def KRS_ABC_func(problem, num_samples, epsilon, ksi, S0, delta_S,
                 verbose=False, save=True):

    M = 50
    E = 50

    y_star = problem.y_star
    y_dim = problem.y_dim
    #eye = np.identity(problem.y_dim)

    simulator = problem.simulator
    statistics = problem.statistics

    # Prior distribution
    prior = problem.prior
    prior_args = problem.prior_args

    # Proposal distribution
    # NOTE: First arg is always theta.
    proposal = problem.proposal
    proposal_args = problem.proposal_args
    use_log = problem.use_log

    theta = problem.theta_init
    if use_log:
        log_theta = np.log(theta)

    samples = np.zeros(num_samples)

    xs = [prior.rvs(*prior_args) for s in xrange(S0)]
    ts = [statistics(simulator(x)) for x in xs]

    print np.array(ts)

    current_sim_calls = S0
    sim_calls = []
    accepted = []

    h = np.var(xs) * pow(3.0 / (4 * len(xs)), 0.2) / 8.0
    h = 1

    #rng = np.linspace(problem.rng[0], problem.rng[1])

    for i in xrange(num_samples):
        if verbose:
            if i % 200 == 0:
                print i

        # Sample theta_p from proposal
        numer = -np.inf
        while np.isneginf(numer):
            theta_p = proposal.rvs(theta, *proposal_args)
            if use_log:
                log_theta_p = np.log(theta_p)

            # Compute alpha using eq. 19
            numer = prior.logpdf(theta_p, *prior_args)
            denom = prior.logpdf(theta, *prior_args)



        if use_log:
            numer += proposal.logpdf(theta, log_theta_p, *proposal_args)
            denom += proposal.logpdf(theta_p, log_theta, *proposal_args)
        else:
            numer += proposal.logpdf(theta, theta_p, *proposal_args)
            denom += proposal.logpdf(theta_p, theta, *proposal_args)

        while True:
            # Get mus and sigmas from Kernel regression
            xs_a = np.array(xs, ndmin=2).T
            ts_a = np.array(ts, ndmin=2).T

            mu_bar = np.zeros(y_dim)
            mu_bar_p = np.zeros(y_dim)
            std = np.zeros(y_dim)
            std_p = np.zeros(y_dim)
            N = np.zeros(y_dim)
            N_p = np.zeros(y_dim)
            cov = dict()
            cov_p = dict()
            mu = dict()
            mu_p = dict()

            for j in xrange(y_dim):
                mu_bar[j], std[j], _, N[j], cov[j] = kernel_regression(
                    np.array(theta), xs_a, ts_a[j, :], h)
                mu_bar_p[j], std_p[j], _, N_p[j], cov_p[j] = kernel_regression(
                    np.array(theta_p), xs_a, ts_a[j, :], h)

                # Get samples
                mu[j] = distr.normal.rvs(mu_bar[j],
                                         std[j] / np.sqrt(N[j]), M)
                mu_p[j] = distr.normal.rvs(mu_bar_p[j],
                                           std_p[j] / np.sqrt(N_p[j]), M)

            alphas = np.zeros(M)
            for m in xrange(M):
                other_term = 0
                for j in xrange(y_dim):
                    other_term += \
                        distr.normal.logpdf(
                            y_star[j],
                            mu_p[j][m],
                            std_p[j] + (epsilon ** 2)) - \
                        distr.normal.logpdf(
                            y_star[j],
                            mu[j][m],
                            std[j] + (epsilon ** 2))

                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            error = np.mean([e * conditional_error(alphas, e, tau, M)
                            for e in np.linspace(0, 1, E)])

            if error > ksi:
                # Acquire training points
                for s in xrange(delta_S):
                    new_x = problem.prior.rvs(*prior_args)
                    ts.append(statistics(simulator(new_x)))
                    xs.append(new_x)
                    current_sim_calls += 1
            else:
                break

        if distr.uniform.rvs() <= tau:
            accepted.append(True)
            theta = theta_p
            if use_log:
                log_theta = log_theta_p
        else:
            accepted.append(False)

        samples[i] = theta
        sim_calls.append(current_sim_calls)

        current_sim_calls = 0

    if save:
        dm.save(KRS_ABC,
                [epsilon, ksi, S0, delta_S],
                problem,
                (samples, sim_calls, accepted))

    return samples, sim_calls, accepted

if __name__ == '__main__':
    from problems import wilkinson_problem
    from compare import variation_distance

    problem = wilkinson_problem()
    samples, sim_calls, accepted = KRS_ABC(
        problem, 10000, 0.05, 0.1, 30, 10, True, save=False)

    rng = np.linspace(problem.rng[0], problem.rng[1], 200)
    plt.hist(samples, bins=100, normed=True)
    plt.show()

    plt.hist(samples, bins=100, normed=True)
    post = problem.true_posterior
    post_args = problem.true_posterior_args
    plt.plot(rng, post.pdf(rng, *post_args))
    plt.show()

    plt.plot(variation_distance(samples, problem))
    plt.show()
