'''
Implements different algorithms for Approximate Bayesian
Computation.

@author Steven
'''

import sys
import numpy as np
from numpy import linalg
import kernels
import kernel_regression as kr
from abc import ABCMeta, abstractmethod
import distributions as distr
import data_manipulation as dm
from utils import logsumexp, conditional_error
from problems import ABC_Problem
import matplotlib.pyplot as plt

__all__ = ['Base_ABC_Algorithm', 'Base_MCMC_ABC_Algorithm',
           'Reject_ABC', 'Marginal_ABC', 'Pseudo_Marginal_ABC',
           'SL_ABC', 'ASL_ABC']


class Base_ABC_Algorithm(object):

    '''
    Abstract base class for ABC algorithms.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, num_samples, verbose=False, save=True,
                 **kwargs):
        self.problem = problem
        self.num_samples = num_samples

        self.y_star = problem.y_star
        self.y_dim = problem.y_dim

        self.simulator = problem.simulator

        self.prior = problem.prior
        self.prior_args = problem.prior_args

        self.statistics = problem.statistics

        self.samples = []
        self.sim_calls = []

        self.verbose = verbose
        self.save = save

        self.needed_params = []

        self.current_sim_calls = 0

    def __repr__(self):
        s = type(self.problem).__name__ + '_' + type(self).__name__
        for par in self.needed_params:
            s += '_' + str(self.__dict__[par])

        return s

    def get_parameters(self):
        '''
        Returns the list of parameter values. Order is determined by
        `self.needed_params`.
        '''
        return [self.__dict__[par] for par in self.needed_params]

    def save_results(self):
        '''
        Saves the results of this algorithm.

        Note: Should be called after a call to `run()`
        '''
        dm.save(self)

    def verbosity(self, i, interval=200):
        if self.verbose and i % interval == 0:
            sys.stdout.write('\r%s iteration %d %d' %
                             (type(self).__name__, i, sum(self.sim_calls)))
            sys.stdout.flush()

    @abstractmethod
    def run(self):
        return NotImplemented


class Reject_ABC(Base_ABC_Algorithm):

    '''
    A simple rejection sampler.
    '''

    def __init__(self, problem, num_samples, epsilon, **kwargs):
        '''
        Creates an instance of rejection ABC for the given problem.

        Parameters
        ----------
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            num_samples : int
                The number of samples to sample.
            epsilon : float
                The error margin or epsilon-tube.
            verbose : bool
                If set to true iteration number as well as number of
                simulation calls will be printed.
            save : bool
                If True will save the result to a (possibly exisisting)
                database
        '''
        super(Reject_ABC, self).__init__(problem, num_samples, **kwargs)

        self.needed_params = ['epsilon']
        self.epsilon = epsilon

    def run(self):
        for i in xrange(self.num_samples):
            self.verbosity(i)

            self.current_sim_calls = 0
            error = self.epsilon + 1.0

            while error > self.epsilon:
                # Sample x from the (uniform) prior
                x = self.prior.rvs(*self.prior_args)

                # Perform simulation
                y = self.statistics(self.simulator(x))
                self.current_sim_calls += 1

                # Calculate error
                # TODO: Implement more comparison methods
                error = linalg.norm(self.y_star - y)

            # Accept the sample
            self.samples.append(x)
            self.sim_calls.append(self.current_sim_calls)

        # Print a newline
        if self.verbose:
            print ''

        if self.save:
            self.save_results()


class Base_MCMC_ABC_Algorithm(Base_ABC_Algorithm):

    '''
    Abstract base class for MCMC ABC algorithms.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, num_samples, verbose=False, save=True,
                 **kwargs):
        super(Base_MCMC_ABC_Algorithm, self).__init__(
            problem, num_samples, verbose, save)

        assert isinstance(problem, ABC_Problem), \
            'Problem is not an instance of ABC_Problem'

        self.proposal = problem.proposal
        self.proposal_args = problem.proposal_args
        self.use_log = problem.use_log

        self.theta = problem.theta_init
        if self.use_log:
            self.log_theta = np.log(self.theta)

        self.accepted = []

    @abstractmethod
    def mh_step(self):
        '''
        The Metropolis-Hastings step of the Markov Chain.
        In this function the acceptance probability will be calculated and
        the sample will be either rejected or accepted.

        Returns
        -------
        sample_accepted : bool
            Whether the sample was accepted
        '''
        return NotImplemented

    def run(self):
        '''
        Runs the algorithm.
        '''
        for i in xrange(self.num_samples):
            # Print information if needed
            self.verbosity(i)

            # Sample theta_p from proposal
            if self.use_log:
                self.theta_p = self.proposal.rvs(
                    self.log_theta, *self.proposal_args)
                self.log_theta_p = np.log(self.theta_p)
            else:
                self.theta_p = self.proposal.rvs(
                    self.theta, *self.proposal_args)

            # Calculate stationary parts of the acceptance probability
            self.prior_logprob_p = self.prior.logpdf(
                self.theta_p, *self.prior_args)
            self.prior_logprob = self.prior.logpdf(
                self.theta, *self.prior_args)

            if self.use_log:
                self.proposal_logprob = self.proposal.logpdf(
                    self.theta, self.log_theta_p, *self.proposal_args)
                self.proposal_logprob_p = self.proposal.logpdf(
                    self.theta_p, self.log_theta, *self.proposal_args)
            else:
                self.proposal_logprob = self.proposal.logpdf(
                    self.theta, self.theta_p, *self.proposal_args)
                self.proposal_logprob_p = self.proposal.logpdf(
                    self.theta_p, self.theta, *self.proposal_args)

            # Perform the Metropolis-Hastings step
            sample_accepted = self.mh_step()

            if sample_accepted:
                self.theta = self.theta_p
                if self.use_log:
                    self.log_theta = self.log_theta_p

            self.accepted.append(sample_accepted)
            self.sim_calls.append(self.current_sim_calls)
            self.samples.append(self.theta)

            # Reset the number of simulation calls for the next iteration
            self.current_sim_calls = 0

        # Print a blank line if needed
        if self.verbose:
            print ''

        # Store the results if necessary
        if self.save:
            self.save_results()


class Marginal_ABC(Base_MCMC_ABC_Algorithm):

    def __init__(self, problem, num_samples, **params):
        '''
        Creates an instance of the marginal ABC algorithm described by Meeds
        and Welling [1]_.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        num_samples : int
            The number of samples to draw.
        epsilon : float
            Error margin.
        S : int
            Number of simulations per iteration.

        Optional Arguments
        ------------------
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulation calls. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database.
            Default `True`.

        References
        ----------
        .. [1] GPS-ABC: Gaussian Process Surrogate Approximate Bayesian
           Computation. E. Meeds and M. Welling.
           http://arxiv.org/abs/1401.2838
        '''

        super(Marginal_ABC, self).__init__(problem, num_samples, **params)

        self.needed_params = ['epsilon', 'S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S = params['S']
        self.epsilon = params['epsilon']

    def mh_step(self):
        diff = []
        diff_p = []

        # Get S samples and approximate marginal likelihood
        for s in xrange(self.S):
            new_x = self.statistics(self.simulator(self.theta))
            new_x_p = self.statistics(self.simulator(self.theta_p))
            self.current_sim_calls += 2

            # Compute the P(y | x, theta) for these samples
            u = linalg.norm(new_x - self.y_star) / self.epsilon
            u_p = linalg.norm(new_x_p - self.y_star) / self.epsilon

            diff.append(kernels.log_gaussian(u / self.epsilon))
            diff_p.append(kernels.log_gaussian(u_p / self.epsilon))

        diff_term = logsumexp(np.array(diff_p)) - logsumexp(np.array(diff))

        # Calculate acceptance according to eq. 4
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        log_alpha = min(0.0, (numer - denom) + diff_term)

        # Accept proposal with probability alpha
        return distr.uniform.rvs(0, 1) <= np.exp(log_alpha)


class Pseudo_Marginal_ABC(Base_MCMC_ABC_Algorithm):

    def __init__(self, problem, num_samples, **params):
        '''
        Creates an instance of the pseudo-marginal ABC algorithm described by
        Meeds and Welling [1]_.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        epsilon : float
            Error margin.
        S : int
            Number of simulations per iteration.
        num_samples : int
            The number of samples to draw.

        Optional Arguments
        ------------------
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulation calls. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database.
            Default `True`.

        References
        ----------
        .. [1] GPS-ABC: Gaussian Process Surrogate Approximate Bayesian
           Computation. E. Meeds and M. Welling.
           http://arxiv.org/abs/1401.2838
        '''
        super(Pseudo_Marginal_ABC, self).__init__(problem, num_samples,
                                                  **params)
        self.needed_params = ['epsilon', 'S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S = params['S']
        self.epsilon = params['epsilon']

        prev_diff = []

        for s in xrange(self.S):
            new_x = self.statistics(self.simulator(self.theta))

            # Compute the P(y | x, theta) for these samples
            u = linalg.norm(new_x - self.y_star) / self.epsilon
            prev_diff.append(kernels.log_gaussian(u / self.epsilon))

        self.prev_diff_term = logsumexp(np.array(prev_diff))
        self.cur_sim_calls = self.S

    def mh_step(self):
        diff_p = []

        # Get S samples and approximate marginal likelihood
        for s in xrange(self.S):
            new_x_p = self.statistics(self.simulator(self.theta_p))

            # Compute the P(y | x, theta) for these samples
            u_p = linalg.norm(new_x_p - self.y_star) / self.epsilon
            diff_p.append(kernels.log_gaussian(u_p / self.epsilon))

        self.cur_sim_calls += self.S

        # Calculate acceptance according to eq. 4
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        cur_diff_term = logsumexp(np.array(diff_p))
        diff_term = cur_diff_term - self.prev_diff_term

        log_alpha = min(0.0, (numer - denom) + diff_term)

        # Accept proposal with probability alpha
        return distr.uniform.rvs(0, 1) <= np.exp(log_alpha)


class SL_ABC(Base_MCMC_ABC_Algorithm):

    def __init__(self, problem, num_samples, **params):
        '''
        Creates an instance of the Synthetic Likelihood ABC algorithm described
        by Meeds and Welling [1]_.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        num_samples : int
            The number of samples
        S : int
            Number of simulations per iteration
        epsilon : float
            Error margin.

        Optional Arguments
        ------------------
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulations. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database
            Default `True`.

        Note that while S and epsilon are keyword-arguments, they are necessary

        References
        ----------
        .. [1] GPS-ABC: Gaussian Process Surrogate Approximate Bayesian
           Computation. E. Meeds and M. Welling.
           http://arxiv.org/abs/1401.2838
        '''
        super(SL_ABC, self).__init__(problem, num_samples, **params)

        self.needed_params = ['epsilon', 'S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S = params['S']
        self.epsilon = params['epsilon']

        self.eps_eye = np.identity(self.y_dim) * self.epsilon ** 2

    def mh_step(self):
        # Get S samples from simulator
        x = np.array([self.statistics(self.simulator(self.theta))
                      for s in xrange(self.S)])
        x_p = np.array([self.statistics(self.simulator(self.theta_p))
                        for s in xrange(self.S)])
        self.current_sim_calls = 2 * self.S

        # Set mu's according to eq. 5
        mu_theta = np.mean(x, 0)
        mu_theta_p = np.mean(x_p, 0)

        # Set sigma's according to eq. 6
        x_m = x - mu_theta
        x_m_p = x_p - mu_theta_p
        sigma_theta = np.dot(x_m.T, x_m) / float(self.S - 1)
        sigma_theta_p = np.dot(x_m_p.T, x_m_p) / float(self.S - 1)

        # Calculate acceptance according to eq. 10
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        other_term = \
            distr.multivariate_normal.logpdf(
                self.y_star,
                mu_theta_p,
                sigma_theta_p + self.eps_eye) - \
            distr.multivariate_normal.logpdf(
                self.y_star,
                mu_theta,
                sigma_theta + self.eps_eye)

        log_alpha = min(0.0, (numer - denom) + other_term)

        return distr.uniform.rvs(0.0, 1.0) <= np.exp(log_alpha)


class ASL_ABC(Base_MCMC_ABC_Algorithm):

    def __init__(self, problem, num_samples, **params):
        '''
        Creates an instance of the Adaptive Synthetic Likelihood ABC algorithm
        described by Meeds and Welling [1]_.

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
        super(ASL_ABC, self).__init__(problem, num_samples, **params)

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

        self.eye_eps = np.identity(self.y_dim) * self.epsilon ** 2

    def mh_step(self):
        # Reset the samples
        x = []
        x_p = []

        additional = self.S0

        while True:
            # Get additional samples from simulator
            x.extend([self.statistics(self.simulator(self.theta))
                      for s in xrange(additional)])
            x_p.extend([self.statistics(self.simulator(self.theta_p))
                       for s in xrange(additional)])

            additional = self.delta_S
            S = len(x)

            # Convert to arrays
            ax = np.array(x)
            ax_p = np.array(x_p)

            # Set mu's according to eq. 5
            mu_hat_theta = np.array(np.mean(ax, 0), ndmin=1)
            mu_hat_theta_p = np.array(np.mean(ax_p, 0), ndmin=1)

            # Set sigma's according to eq. 6
            # TODO: incremental implementation?
            x_m = ax - mu_hat_theta
            x_m_p = ax_p - mu_hat_theta_p
            sigma_theta = np.array(np.dot(x_m.T, x_m) / float(S - 1),
                                   ndmin=2)
            sigma_theta_p = np.array(np.dot(x_m_p.T, x_m_p) / float(S - 1),
                                     ndmin=2)

            sigma_theta_S = sigma_theta / float(S)
            sigma_theta_p_S = sigma_theta_p / float(S)

            alphas = np.zeros(self.M)
            for m in range(self.M):
                # Sample mu_theta_p and mu_theta using eq. 11
                mu_theta = distr.multivariate_normal.rvs(mu_hat_theta,
                                                         sigma_theta_S)
                mu_theta_p = distr.multivariate_normal.rvs(mu_hat_theta_p,
                                                           sigma_theta_p_S)

                # Compute alpha using eq. 12
                numer = self.prior_logprob_p + self.proposal_logprob + \
                    distr.multivariate_normal.logpdf(
                        self.y_star,
                        mu_theta_p,
                        sigma_theta_p + self.eye_eps)

                denom = self.prior_logprob + self.proposal_logprob_p + \
                    distr.multivariate_normal.logpdf(
                        self.y_star,
                        mu_theta,
                        sigma_theta + self.eye_eps)

                log_alpha = min(0.0, numer - denom)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate
            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error < self.ksi:
                break

        self.current_sim_calls = 2 * S

        return distr.uniform.rvs() <= tau


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

        self.eps_eye = np.identity(self.y_dim) * self.epsilon ** 2

        # TODO: Set this more intelligently
        self.h = 0.2

        # Initialize the surrogate with S0 samples from the prior
        self.xs = [self.prior.rvs(*self.prior_args) for s in xrange(self.S0)]
        self.ts = [self.statistics(self.simulator(x)) for x in self.xs]

        self.current_sim_calls = self.S0
        self.y_star = np.array(self.y_star, ndmin=1)

    def mh_step(self):

        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        mu_bar = np.zeros(self.y_dim)
        mu_bar_p = np.zeros(self.y_dim)
        std = np.zeros(self.y_dim)
        std_p = np.zeros(self.y_dim)
        mu_S = np.zeros(self.y_dim)
        mu_S_p = np.zeros(self.y_dim)

        while True:
            # Turn the lists into arrays
            xs_a = np.array(self.xs, ndmin=2)
            ts_a = np.array(self.ts, ndmin=2)

            # Get statistics from surrogate
            for j in xrange(self.y_dim):

                #rng = np.linspace(0, 2 * np.pi, 100)

                #rng = np.linspace(self.problem.rng[0], self.problem.rng[1])
                #plot_krs(xs_a, ts_a[:, [j]], self.h, rng, self.y_star[j])
                #plt.show()

                tup = kr.kernel_regression(
                    self.theta, xs_a, ts_a[:, [j]], self.h)
                tup_p = kr.kernel_regression(
                    self.theta_p, xs_a, ts_a[:, [j]], self.h)

                mu_bar[j] = tup[0]
                mu_bar_p[j] = tup[0]

                std[j] = tup_p[1]
                std_p[j] = tup_p[1]

                mu_S[j] = tup[1] / np.sqrt(tup[3])
                mu_S_p[j] = tup_p[1] / np.sqrt(tup_p[3])

            alphas = np.zeros(self.M)
            for m in xrange(self.M):
                other_term = numer - denom

                for j in xrange(self.y_dim):
                    mu = distr.normal.rvs(mu_bar[j], mu_S[j])
                    mu_p = distr.normal.rvs(mu_bar_p[j], mu_S_p[j])

                    other_term += distr.normal.logpdf(
                        self.y_star[j], mu_p, std_p[j])
                    other_term -= distr.normal.logpdf(
                        self.y_star[j], mu, std[j])

                log_alpha = min(0.0, other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate
            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error < self.ksi:
                break
            else:
                for s in xrange(self.delta_S):
                    # Acquire training points
                    new_x = self.prior.rvs(*self.prior_args)
                    self.ts.append(self.statistics(self.simulator(new_x)))
                    self.xs.append(new_x)
                self.current_sim_calls += self.delta_S

        return distr.uniform.rvs() <= tau


def plot_krs(xs, ts, h, rng, y_star):
    means = np.zeros(len(rng))
    stds = np.zeros(len(rng))
    Ns = np.zeros(len(rng))
    confs = np.zeros(len(rng))
    for i, val in enumerate(rng):
        means[i], stds[i], confs[i], Ns[i], _ = \
            kr.kernel_regression(val, xs, ts, h)

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
    plt.plot(rng, means)
    plt.scatter(xs, ts)
    plt.axhline(y_star)
    plt.ylim(-4, 4)
    plt.title('S = {0}, h = {1}'.format(len(xs), h))
