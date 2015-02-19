'''
Implements different algorithms for Approximate Bayesian
Computation.

@author Steven
'''

import sys
import numpy as np
import scipy.spatial as sp
from numpy import linalg
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

import kernels
import kernel_methods as km
import distributions as distr
import data_manipulation as dm
from utils import logsumexp, conditional_error, \
    get_bootstrap, get_weighted_bootstrap, get_bootstrap_ids, \
    lowessNd
from problems import ABC_Problem

__all__ = ['Base_ABC',
           'Base_MCMC_ABC',
           'Reject_ABC',
           'Marginal_ABC',
           'Pseudo_Marginal_ABC',
           'Base_SL_ABC',
           'SL_ABC',
           'KL_ABC',
           'ASL_ABC',
           'AKL_ABC',
           'PKS_ABC',
           'PSS_ABC']


class Base_ABC(object):

    '''
    Abstract base class for ABC algorithms.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, verbose=False, save=True, data_dir=True,
                 **kwargs):

        self.problem = problem

        self.y_star = problem.y_star
        self.y_dim = problem.y_dim

        self.simulator = problem.simulator

        self.prior = problem.prior
        self.prior_args = problem.prior_args

        self.statistics = problem.statistics

        self.verbose = verbose
        self.save = save

        self.needed_params = []

        self.current_sim_calls = 0

        self.samples = []
        self.stats = []
        self.sim_calls = []

        self.just_reset = False

        self.data_dir = data_dir

    def __repr__(self):
        # This is used to create a data file for the results
        s = type(self.problem).__name__ + '_' + type(self).__name__
        for par in self.needed_params:
            s += '_' + str(self.__dict__[par])

        return s

    def reset(self):
        '''
        Resets the internal lists. So that an new run from scratch can begin.
        '''

        self.current_sim_calls = 0

        self.samples = []
        self.stats = []
        self.sim_calls = []

        self.just_reset = True

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
        dm.save(self, data_dir=self.data_dir)

    def verbosity(self, i, interval=10):
        if self.verbose and i % interval == 0:
            sys.stdout.write('\r%s iteration %d %d' %
                             (type(self).__name__, i, sum(self.sim_calls)))
            sys.stdout.flush()

    @abstractmethod
    def run(self, num_samples, reset=True):
        return NotImplemented


class Reject_ABC(Base_ABC):

    '''
    A simple rejection sampler.
    '''

    def __init__(self, problem, epsilon, **kwargs):
        '''
        Creates an instance of rejection ABC for the given problem.

        Parameters
        ----------
            problem : ABC_Problem instance
                The problem to solve An instance of the ABC_Problem class.
            epsilon : float
                The error margin or epsilon-tube.
            verbose : bool
                If set to true iteration number as well as number of
                simulation calls will be printed.
            save : bool
                If True will save the result to a (possibly exisisting)
                database
        '''
        super(Reject_ABC, self).__init__(problem, **kwargs)

        self.needed_params = ['epsilon']
        self.epsilon = epsilon

    def run(self, num_samples, reset=True):
        '''
        Runs the algorithm.

        Parameters
        ----------
        num_samples : int
            number of samples to get
        reset : bool
            Whether the internal lists should be reset.
            If false it continues where it stopped.
            Default True.
        '''
        # Reset previous values
        if reset:
            self.reset()

        for i in xrange(num_samples):
            self.verbosity(i)

            self.current_sim_calls = 0
            error = self.epsilon + 1.0

            while error > self.epsilon:
                # Sample theta from the prior
                theta = self.prior.rvs(*self.prior_args)

                # Perform simulation
                y = self.statistics(self.simulator(theta))
                self.current_sim_calls += 1

                # Calculate error
                # TODO: Implement more comparison methods
                error = linalg.norm(self.y_star - y)

                if self.current_sim_calls % 1000 == 0:
                    print self.current_sim_calls, 'sim_calls done'

            # Accept the sample
            self.samples.append(theta)
            self.stats.append(y)
            self.sim_calls.append(self.current_sim_calls)

        # Print a newline
        if self.verbose:
            print ''

        if self.save:
            self.save_results()


class Base_MCMC_ABC(Base_ABC):

    '''
    Abstract base class for MCMC ABC algorithms.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, verbose=False, save=True, data_dir=True, **kwargs):
        super(Base_MCMC_ABC, self).__init__(
            problem, verbose, save, data_dir)

        assert isinstance(problem, ABC_Problem), \
            'Problem is not an instance of ABC_Problem'

        self.proposal = problem.proposal
        self.proposal_args = problem.proposal_args
        self.use_log = problem.use_log

        self.theta_init = problem.get_theta_init()

        self.accepted = []

        self.theta = self.theta_init
        if self.use_log:
            self.log_theta = np.log(self.theta)
        self.just_reset = True

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

    def reset(self):
        super(Base_MCMC_ABC, self).reset()

        self.accepted = []

        self.theta = self.problem.get_theta_init()
        if self.use_log:
            self.log_theta = np.log(self.theta)

    def run(self, num_samples, reset=True):
        '''
        Runs the algorithm.

        Parameters
        ----------
        num_samples : int
            number of samples to get
        reset : bool
            Whether the internal lists should be reset.
            If false it continues where it stopped.
            Default True.
        '''
        # Reset previous values if needed
        if reset and not self.just_reset:
            self.reset()

        self.just_reset = False

        for i in xrange(num_samples):
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


class Marginal_ABC(Base_MCMC_ABC):

    '''
    Marginal ABC
    ------------
    Approximates the likelihood by a Monte Carlo estimate of the integral.
    The marginal sampler re-estimates both the denominator as well as the
    numerator each iteration, which in practice leads to better mixing.
    '''

    def __init__(self, problem, **params):
        '''
        Creates an instance of the marginal ABC algorithm described by Meeds
        and Welling [1]_.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
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

        super(Marginal_ABC, self).__init__(problem, **params)

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


class Pseudo_Marginal_ABC(Base_MCMC_ABC):

    '''
    Pseudo Marginal ABC
    -------------------
    Approximates the likelihood by a Monte Carlo estimate of the integral.
    The pseudo marginal sampler only re-estimates the numerator each iteration.
    The denominator is carried over from the previous iteration. This is in
    practice slower in mixing, but achieves lower bias.
    '''

    def __init__(self, problem, **params):
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
        super(Pseudo_Marginal_ABC, self).__init__(problem, **params)
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
        accept = distr.uniform.rvs(0, 1) <= np.exp(log_alpha)

        if accept:
            self.prev_diff_term = cur_diff_term

        return accept


class Base_SL_ABC(Base_MCMC_ABC):

    '''
    Abstract Base class for Synthetic Likelihood ABC.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, **params):
        super(Base_SL_ABC, self).__init__(problem, **params)

        self.needed_params = ['S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S = params['S']

    def mh_step(self):
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        # Early exit
        if np.isneginf(numer):
            return False

        # Get S samples from simulator
        # TODO: Make marginal/pseudo marginal choosable
        self.x = np.array([self.statistics(self.simulator(self.theta))
                           for s in xrange(self.S)], ndmin=2)
        self.x_p = np.array([self.statistics(self.simulator(self.theta_p))
                             for s in xrange(self.S)], ndmin=2)
        self.current_sim_calls = 2 * self.S

        other_term = self.get_SL_estimate()

        log_alpha = min(0.0, (numer - denom) + other_term)

        return distr.uniform.rvs(0.0, 1.0) <= np.exp(log_alpha)

    @abstractmethod
    def get_SL_estimate(self):
        '''
        Returns an estimate of p(y_star | theta_p) / p(y_star | theta)
        in log space.
        '''
        return NotImplemented


class SL_ABC(Base_SL_ABC):

    '''
    Synthetic Likelihood ABC
    ------------------------
    Approximates the likelihood with a normal distribution, by estimating
    the first and second order statistics from samples from the simulator.
    '''

    def __init__(self, problem, **params):
        '''
        Creates an instance of the Synthetic Likelihood ABC algorithm described
        by Meeds and Welling [1]_.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
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
        super(SL_ABC, self).__init__(problem, **params)

        self.epsilon = 0.0
        if 'epsilon' in params.keys():
            self.epsilon = params['epsilon']

        if 'diag' in params.keys():
            self.diag = params['diag']

        self.eps_eye = np.identity(self.y_dim) * self.epsilon ** 2

    def __repr__(self):
        name = super(SL_ABC, self).__repr__() + '_' + str(self.epsilon)
        if self.diag:
            return name + '_diag'
        return name

    def get_SL_estimate(self):

        # Set mu's according to eq. 5
        mu_theta = np.mean(self.x, 0)
        mu_theta_p = np.mean(self.x_p, 0)

        # Set sigma's according to eq. 6
        x_m = self.x - mu_theta
        x_m_p = self.x_p - mu_theta_p
        sigma_theta = np.dot(x_m.T, x_m) / float(self.S - 1)
        sigma_theta_p = np.dot(x_m_p.T, x_m_p) / float(self.S - 1)

        if self.diag:
            sigma_theta = np.diag(np.diag(sigma_theta))
            sigma_theta_p = np.diag(np.diag(sigma_theta_p))

        other_term = \
            distr.multivariate_normal.logpdf(
                self.y_star,
                mu_theta_p,
                sigma_theta_p + self.eps_eye) - \
            distr.multivariate_normal.logpdf(
                self.y_star,
                mu_theta,
                sigma_theta + self.eps_eye)

        return other_term


class KL_ABC(Base_SL_ABC):

    '''
    KDE Likelihood ABC
    ------------------------
    Approximates the likelihood with a kernel density approximation, by
    estimating the first and second order statistics from samples from the
    simulator.
    '''

    def __init__(self, problem, **params):
        '''
        Creates an instance of the Kernel Synthetic Likelihood ABC algorithm.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        S : int
            Number of simulations per iteration

        Optional Arguments
        ------------------
        kernel : kernel function
            The kernel to use in the x-direction.
            Default Gaussian.
        bandwidth : float or string
            The bandwidth estimation method to use. If `h` is a `float`, it
            will be used as the bandwidth.
            Supported methods are:
            - 'SJ': Sheather-Jones plug-in estimate
            - 'Silverman': Silvermans rule of thumb
            - 'Scott': Scotts rule of thumb
            Default 'Silverman'.
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulations. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database
            Default `True`.

        Note that while S is a keyword-argument, they are necessary
        '''
        super(KL_ABC, self).__init__(problem, **params)

        self.needed_params = ['S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S = params['S']

        if 'kernel' in params.keys():
            self.kernel = params['kernel']
        else:
            self.kernel = kernels.gaussian

        if 'bandwidth' in params.keys():
            self.bandwidth = params['bandwidth']
        else:
            self.bandwidth = 'Silverman'

        if 'adaptive' in params.keys():
            self.adaptive = params['adaptive']
        else:
            self.adaptive = False

        if 'nonradial' in params.keys():
            self.nonradial = params['nonradial']
        else:
            self.nonradial = False

    def __repr__(self):
        name = super(KL_ABC, self).__repr__() + '_' + self.kernel.__name__
        if self.bandwidth != 'Silverman':
             name += '_bw_' + str(self.bandwidth)
        if self.nonradial:
            name += '_nonradial'
        return name

    def get_SL_estimate(self):
        if self.adaptive:
            pass
        else:
            other_term = 0.0

            for j in range(self.y_dim):
                other_term += \
                    km.kernel_density_estimate(
                        self.y_star[j], self.x_p[:, [j]],
                        self.kernel, self.bandwidth, nonradial=self.nonradial) - \
                    km.kernel_density_estimate(
                        self.y_star[j], self.x[:, [j]],
                        self.kernel, self.bandwidth, nonradial=self.nonradial)

        return other_term


class ASL_ABC(Base_MCMC_ABC):

    '''
    Adaptive Synthetic Likelihood ABC
    ------------------------
    Approximates the likelihood with a normal distribution, by estimating
    the first and second order statistics from samples from the simulator.

    However the error on making a mistake with the acceptance probability
    (due to approximation with a finite number of samples) is used to
    determine whether more samples are needed. Hence the number of samples
    that is drawn each iteration is adaptive to the acceptance error.
    '''

    def __init__(self, problem, **params):
        '''
        Creates an instance of the Adaptive Synthetic Likelihood ABC algorithm
        described by Meeds and Welling [1]_.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
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
        super(ASL_ABC, self).__init__(problem, **params)

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
            x_m = ax - mu_hat_theta
            x_m_p = ax_p - mu_hat_theta_p
            sigma_theta = \
                np.array(np.dot(x_m.T, x_m) / float(S - 1), ndmin=2)
            sigma_theta_p = \
                np.array(np.dot(x_m_p.T, x_m_p) / float(S - 1), ndmin=2)

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


class AKL_ABC(Base_MCMC_ABC):

    '''
    Adaptive Kernel Synthetic Likelihood ABC
    ----------------------------------------
    Approximates the likelihood with a kernel density estimate.

    However the error on making a mistake with the acceptance probability
    (due to approximation with a finite number of samples) is used to
    determine whether more samples are needed. Hence the number of samples
    that is drawn each iteration is adaptive to the acceptance error.

    The error is estimated using bootstrapping.
    '''

    def __init__(self, problem, **params):
        '''
        Creates an instance of the Adaptive KDE Likelihood ABC algorithm.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
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
            Number of bootstrap repetitions. Default 100.
        E : int
            Number of points to approximate conditional error. Default 50.
        '''
        super(AKL_ABC, self).__init__(problem, **params)

        self.needed_params = ['ksi', 'S0', 'delta_S']
        assert set(self.needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(self.needed_params))

        self.S0 = params['S0']
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

        if 'kernel' in params.keys():
            self.kernel = params['kernel']
        else:
            self.kernel = kernels.gaussian

        if 'bandwidth' in params.keys():
            self.bandwidth = params['bandwidth']
        else:
            self.bandwidth = 'Silverman'

    def __repr__(self):
        return super(AKL_ABC, self).__repr__() + '_' + self.kernel.__name__

    def mh_step(self):
        # Reset the samples
        x = []
        x_p = []

        additional = self.S0

        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        if np.isneginf(numer):
            return False

        while True:
            # Get additional samples from simulator
            x.extend([self.statistics(self.simulator(self.theta))
                      for s in xrange(additional)])
            x_p.extend([self.statistics(self.simulator(self.theta_p))
                       for s in xrange(additional)])

            additional = self.delta_S
            S = len(x)

            alphas = np.zeros(self.M)
            for m in xrange(self.M):
                # Get a bootstrap sample
                new_x = get_bootstrap(x)
                new_x_p = get_bootstrap(x_p)

                # Compute alpha using eq. 12
                other_term = 0.0

                for j in range(self.y_dim):
                    other_term += \
                        km.kernel_density_estimate(
                                self.y_star[j], new_x_p[:, [j]],
                                self.kernel, self.bandwidth) - \
                        km.kernel_density_estimate(
                                self.y_star[j], new_x[:, [j]],
                                self.kernel, self.bandwidth)

                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate
            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error < self.ksi:
                break

        self.current_sim_calls = 2 * S

        return distr.uniform.rvs() <= tau


class PSS_ABC(Base_MCMC_ABC):

    '''
    Projected Synthetic Surrogate - ABC
    -----------------------------
    Approximates the simulator using a kernel weights on the x and a Gaussian
    approximation on the y, while using a linear regression to correct
    projection.
    '''

    def __init__(self, problem,
                 S0,
                 delta_S,
                 ksi=None,
                 S=None,
                 kernel_x=kernels.gaussian,
                 h_x_method='Silverman',
                 epsilon=0.0,
                 kernel_lowess=kernels.tricube,
                 M=50,
                 E=50,
                 use_tree=False,
                 adaptive=False,
                 alpha=0.5,
                 smoothed_bs=True,
                 orthogonal=False,
                 verbose=False,
                 save=True):
        '''
        Creates an instance of the Linear Corrected Kernel Surrogate ABC
        algorithm.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        S0 : int
            Number of initial simulations
        delta_S : int
            Number of additional simulations, if needed

        Note that while ksi and S are keyword-arguments, one of them is
        necessary.

        Optional Arguments
        ------------------
        kernel_x : kernel function
            The kernel to use in the x-direction.
            Default Gaussian.
        h_x_method : float or string
            The method to estimate the bandwidth with in the x direction.
            Default Silvermans rule of thumb.
        kernel_y : kernel function
            The kernel to use in the y-direction.
            Default Gaussian.
        h_y_method : float or string
            The method to estimate the bandwidth with in the y direction.
            Default Silvermans rule of thumb.
        kernel_lowess : kernel function
            The kernel to use for the local linear regression.
            Default Tricube.
        M : int
            Number of bootstrap repetitions. Default 50.
        E : int
            Number of points to approximate conditional error. Default 50.
        use_tree : bool
            Whether to use a KD tree to make KDE faster.
        adaptive : bool
            Whether to use adaptive kernel density estimates. AKDE works
            better in higher dimensions.
            Default `False`.
        alpha : float
            The sensivity parameter for the sample point adaptive estimator.
            Only used when adaptive is set to `True`.
            Default 0.5.
        ksi : float or `None`
            Error margin on the acceptance probability. This is an uncertainty
            criterion.
            Default `None`.
        S : float or `None`
            Minimal number of samples to have on each location. Note that this
            is another uncertainty criterion
        smoothed_bs : bool
            Whether to use smoothed bootstrapping. Default `True`
        orthogonal : bool
            Whether to use orthogonal projection or linear correction.
            Default `False` (i.e. Linear Correction).
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulations. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database
            Default `True`.
        '''
        super(PSS_ABC, self).__init__(problem, verbose=verbose, save=save)

        assert S is not None or ksi is not None, \
            'There needs to be an uncertainty measure: either S or ksi'

        self.needed_params = ['S0', 'delta_S', 'ksi', 'S']

        self.S0 = S0
        self.delta_S = delta_S
        self.ksi = ksi

        self.kernel_x = kernel_x
        self.h_x_method = h_x_method
        self.kernel_lowess = kernel_lowess
        self.M = M
        self.E = E
        self.use_tree = use_tree
        self.alpha = alpha
        self.adaptive = adaptive
        self.epsilon = epsilon
        self.eps_sqr = self.epsilon ** 2
        self.S = S
        self.smoothed_bs = smoothed_bs
        self.orthogonal = orthogonal

        # Initialize the surrogate with S0 samples from the prior
        self.xs = [self.prior.rvs(*self.prior_args) for i in range(self.S0)]
        self.ts = [self.statistics(self.simulator(x)) for x in self.xs]

        self.current_sim_calls = self.S0

        xs_a = np.array(self.xs, ndmin=2)

        if self.S is not None:
            self.threshold = self.S
        else:
            self.threshold = 0.0


        if self.use_tree:
            self.tree = sp.KDTree(xs_a)
        else:
            self.tree = None

        if self.adaptive:
            log_ind_h, self.dist, self.h_x = \
                km.sample_point_adaptive_weights(xs_a,
                                                 self.kernel_x,
                                                 self.h_x_method,
                                                 self.alpha)
            self.ind_h = np.exp(log_ind_h)
        else:
            self.h_x = km.set_bandwidth(self.h_x_method, xs_a)
            self.ind_h = None

        self.y_star = np.array(self.y_star, ndmin=1)

    def reset(self):
        super(PSS_ABC, self).reset()

        # Initialize the surrogate with S0 samples from the prior
        self.xs = [np.array(self.prior.rvs(*self.prior_args), ndmin=1)
                   for s in xrange(self.S0)]
        self.ts = [self.statistics(self.simulator(x)) for x in self.xs]
        self.current_sim_calls = self.S0

    def get_samples(self):
        for s in xrange(self.delta_S):
            if distr.uniform.rvs() > 0.5:
                new_x = self.theta
            else:
                new_x = self.theta_p

            #new_x = self.prior.rvs(*self.prior_args)

            self.ts.append(self.statistics(self.simulator(new_x)))
            self.xs.append(new_x)

            if self.adaptive:
                # Update distances matrix
                self.dist = km.update_dist(new_x, self.xs, self.dist)

        self.current_sim_calls += self.delta_S

        xs_a = np.array(self.xs, ndmin=2)

        if self.use_tree:
            self.tree = sp.KDTree(xs_a)

        if self.adaptive:
            # Update individual bandwidths
            log_lambda, _, self.h_x = \
                km.sample_point_adaptive_weights(
                    xs_a,
                    self.kernel_x,
                    self.h_x_method,
                    self.alpha,
                    self.dist)
            self.ind_h = np.exp(log_lambda)
        else:
            self.h_x = km.set_bandwidth(self.h_x_method, xs_a)

    def __repr__(self):
        name = super(PSS_ABC, self).__repr__() + \
                '_kx_' + self.kernel_x.__name__

        if self.h_x_method != 'Silverman':
            name += '_hx_' + str(self.h_x_method)

        if self.orthogonal:
            name += '_orth'
        return name


    def mh_step(self):
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        # Shortcut
        if np.isneginf(numer):
            return False

        while True:

            # Turn the lists into arrays of the right shape
            xs_a = np.array(self.xs, ndmin=2)
            ts_a = np.array(self.ts, ndmin=2)

            if self.use_tree:
                dist_x = None
                dist_x_p = None
            else:
                dist_x = km.calc_dist(self.theta, xs_a)
                dist_x_p = km.calc_dist(self.theta_p, xs_a)

            weights_x = \
                km.kernel_weights(self.theta,
                                  xs_a,
                                  self.kernel_x,
                                  self.h_x,
                                  ind_h=self.ind_h,
                                  dist=dist_x,
                                  tree=self.tree)
            weights_x_p = \
                km.kernel_weights(self.theta_p,
                                  xs_a,
                                  self.kernel_x,
                                  self.h_x,
                                  ind_h=self.ind_h,
                                  dist=dist_x_p,
                                  tree=self.tree)

            swx = np.sum(weights_x)
            swx_p = np.sum(weights_x_p)

            if swx <= self.threshold or swx_p <= self.threshold:
                # If at either location we don't have enough samples,
                # Get additional samples, so we can base the
                # decision on something.
                self.get_samples()
                continue

            sheared_y = dict()
            sheared_y_p = dict()
            mu = dict()
            mu_p = dict()
            std = dict()
            std_p = dict()

            for j in xrange(self.y_dim):
                if self.orthogonal:
                    # Orthogonal projection
                    beta = np.zeros(xs_a.shape[1] + 1)
                    beta_p = np.zeros(xs_a.shape[1] + 1)
                else:
                    # Find line
                    beta, dist_x = \
                        lowessNd(self.theta,
                                 xs_a,
                                 ts_a[:, [j]],
                                 self.kernel_lowess)
                    beta_p, dist_x_p = \
                        lowessNd(self.theta_p,
                                 xs_a,
                                 ts_a[:, [j]],
                                 self.kernel_lowess)

                # Translate and shear
                x_t = xs_a - self.theta
                x_t_p = xs_a - self.theta_p

                sheared_y[j] = ts_a[:, [j]] - \
                    np.sum(beta[1:] * x_t, axis=1, keepdims=True)
                sheared_y_p[j] = ts_a[:, [j]] - \
                    np.sum(beta_p[1:] * x_t_p, axis=1, keepdims=True)

                mu_bar = np.sum(sheared_y[j].ravel() * weights_x) / swx
                mu_bar_p = np.sum(sheared_y_p[j].ravel() * weights_x_p) / swx_p

                std[j] = np.sqrt(np.sum(weights_x * (sheared_y[j].ravel() - mu_bar) ** 2.0 ) / swx)
                std_p[j] = np.sqrt(np.sum(weights_x_p * (sheared_y_p[j].ravel() - mu_bar_p) ** 2.0 ) / swx_p)

                # Get samples
                mu[j] = distr.normal.rvs(
                    mu_bar, std[j] / np.sqrt(swx) + 1e-10, self.M)
                mu_p[j] = distr.normal.rvs(
                    mu_bar_p, std_p[j] / np.sqrt(swx_p) + 1e-10, self.M)

            alphas = np.zeros(self.M)
            dim = xs_a.shape[1]

            for m in xrange(self.M):
                other_term = 0.0

                for j in xrange(self.y_dim):
                    other_term += \
                        distr.normal.logpdf(
                            self.y_star[j],
                            mu_p[j][m],
                            std_p[j] + self.eps_sqr) - \
                        distr.normal.logpdf(
                            self.y_star[j],
                            mu[j][m],
                            std[j] + self.eps_sqr)

                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate / grid values
            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error > self.ksi:
                # Acquire training points
                self.get_samples()
            else:
                return distr.uniform.rvs() < tau


class PKS_ABC(Base_MCMC_ABC):

    '''
    Linear Corrected Kernel Surrogate - ABC
    -----------------------------
    Approximates the simulator using a kernel weights on the x and a kernel
    density estimate on the y, while using a linear regression to correct
    projection.
    '''

    def __init__(self, problem,
                 S0,
                 delta_S,
                 ksi=None,
                 S=None,
                 kernel_x=kernels.gaussian,
                 h_x_method='Silverman',
                 kernel_y=kernels.gaussian,
                 h_y_method='Silverman',
                 kernel_lowess=kernels.tricube,
                 M=50,
                 E=50,
                 use_tree=False,
                 adaptive=False,
                 alpha=0.5,
                 smoothed_bs=True,
                 orthogonal=False,
                 verbose=False,
                 save=True,
                 data_dir=True):
        '''
        Creates an instance of the Linear Corrected Kernel Surrogate ABC
        algorithm.

        Parameters
        ----------
        problem : An instance of (a subclass of) `ABC_Problem`.
            The problem to solve.
        S0 : int
            Number of initial simulations
        delta_S : int
            Number of additional simulations, if needed

        Note that while ksi and S are keyword-arguments, one of them is
        necessary.

        Optional Arguments
        ------------------
        kernel_x : kernel function
            The kernel to use in the x-direction.
            Default Gaussian.
        h_x_method : float or string
            The method to estimate the bandwidth with in the x direction.
            Default Silvermans rule of thumb.
        kernel_y : kernel function
            The kernel to use in the y-direction.
            Default Gaussian.
        h_y_method : float or string
            The method to estimate the bandwidth with in the y direction.
            Default Silvermans rule of thumb.
        kernel_lowess : kernel function
            The kernel to use for the local linear regression.
            Default Tricube.
        M : int
            Number of bootstrap repetitions. Default 50.
        E : int
            Number of points to approximate conditional error. Default 50.
        use_tree : bool
            Whether to use a KD tree to make KDE faster.
        adaptive : bool
            Whether to use adaptive kernel density estimates. AKDE works
            better in higher dimensions.
            Default `False`.
        alpha : float
            The sensivity parameter for the sample point adaptive estimator.
            Only used when adaptive is set to `True`.
            Default 0.5.
        ksi : float or `None`
            Error margin on the acceptance probability. This is an uncertainty
            criterion.
            Default `None`.
        S : float or `None`
            Minimal number of samples to have on each location. Note that this
            is another uncertainty criterion
        smoothed_bs : bool
            Whether to use smoothed bootstrapping. Default `True`
        orthogonal : bool
            Whether to use orthogonal projection or linear correction.
            Default `False` (i.e. Linear Correction).
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulations. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database
            Default `True`.
        '''
        super(PKS_ABC, self).__init__(problem, verbose=verbose, save=save, data_dir=data_dir)

        assert S is not None or ksi is not None, \
            'There needs to be an uncertainty measure: either S or ksi'

        self.needed_params = ['S0', 'delta_S', 'ksi', 'S']

        self.S0 = S0
        self.delta_S = delta_S
        self.ksi = ksi

        self.kernel_x = kernel_x
        self.h_x_method = h_x_method
        self.kernel_y = kernel_y
        self.h_y_method = h_y_method
        self.kernel_lowess = kernel_lowess
        self.M = M
        self.E = E
        self.use_tree = use_tree
        self.alpha = alpha
        self.adaptive = adaptive
        self.S = S
        self.smoothed_bs = smoothed_bs
        self.orthogonal = orthogonal

        # Initialize the surrogate with S0 samples from the prior
        self.xs = [self.prior.rvs(*self.prior_args) for i in range(self.S0)]
        self.ts = [self.statistics(self.simulator(x)) for x in self.xs]

        self.current_sim_calls = self.S0

        xs_a = np.array(self.xs, ndmin=2)

        if self.S is not None:
            self.threshold = self.S
        else:
            self.threshold = 0.0


        if self.use_tree:
            self.tree = sp.KDTree(xs_a)
        else:
            self.tree = None

        if self.adaptive:
            log_ind_h, self.dist, self.h_x = \
                km.sample_point_adaptive_weights(xs_a,
                                                 self.kernel_x,
                                                 self.h_x_method,
                                                 self.alpha)
            self.ind_h = np.exp(log_ind_h)
        else:
            self.h_x = km.set_bandwidth(self.h_x_method, xs_a)
            self.ind_h = None

        self.y_star = np.array(self.y_star, ndmin=1)

    def reset(self):
        super(PKS_ABC, self).reset()

        # Initialize the surrogate with S0 samples from the prior
        self.xs = [np.array(self.prior.rvs(*self.prior_args), ndmin=1)
                   for s in xrange(self.S0)]
        self.ts = [self.statistics(self.simulator(x)) for x in self.xs]
        self.current_sim_calls = self.S0

    def get_samples(self):
        for s in xrange(self.delta_S):
            if distr.uniform.rvs() > 0.5:
                new_x = self.theta
            else:
                new_x = self.theta_p

            self.ts.append(self.statistics(self.simulator(new_x)))
            self.xs.append(new_x)

            if self.adaptive:
                # Update distances matrix
                self.dist = km.update_dist(new_x, self.xs, self.dist)

        self.current_sim_calls += self.delta_S

        xs_a = np.array(self.xs, ndmin=2, dtype=float)

        if self.use_tree:
            self.tree = sp.KDTree(xs_a)

        if self.adaptive:
            # Update individual bandwidths
            log_lambda, _, self.h_x = \
                km.sample_point_adaptive_weights(
                    xs_a,
                    self.kernel_x,
                    self.h_x_method,
                    self.alpha,
                    self.dist)
            self.ind_h = np.exp(log_lambda)
        else:
            self.h_x = km.set_bandwidth(self.h_x_method, xs_a)

    def __repr__(self):
        name = super(PKS_ABC, self).__repr__() + \
                '_kx_' + self.kernel_x.__name__ + \
                '_ky_' + self.kernel_y.__name__

        if self.h_x_method != 'Silverman':
            name += '_hx_' + str(self.h_x_method)

        if self.h_y_method != 'Silverman':
            name += '_hy_' + str(self.h_y_method)

        if self.orthogonal:
            name += '_orth'
        if self.smoothed_bs:
            name += '_smooth'
        return name


    def mh_step(self):
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        # Shortcut
        if np.isneginf(numer):
            return False

        while True:
            # Turn the lists into arrays of the right shape
            xs_a = np.array(self.xs, ndmin=2, dtype=float)
            ts_a = np.array(self.ts, ndmin=2, dtype=float)

            if self.use_tree:
                dist_x = None
                dist_x_p = None
            else:
                dist_x = km.calc_dist(self.theta, xs_a)
                dist_x_p = km.calc_dist(self.theta_p, xs_a)

            weights_x = \
                km.kernel_weights(self.theta,
                                  xs_a,
                                  self.kernel_x,
                                  self.h_x,
                                  ind_h=self.ind_h,
                                  dist=dist_x,
                                  tree=self.tree)
            weights_x_p = \
                km.kernel_weights(self.theta_p,
                                  xs_a,
                                  self.kernel_x,
                                  self.h_x,
                                  ind_h=self.ind_h,
                                  dist=dist_x_p,
                                  tree=self.tree)

            swx = np.sum(weights_x)
            swx_p = np.sum(weights_x_p)

            if swx <= self.threshold or swx_p <= self.threshold:
                # If at either location we don't have enough samples,
                # Get additional samples, so we can base the
                # decision on something.
                self.get_samples()
                continue

            sheared_y = dict()
            sheared_y_p = dict()

            for j in xrange(self.y_dim):
                if self.orthogonal:
                    # Orthogonal projection
                    beta = np.zeros(xs_a.shape[1] + 1)
                    beta_p = np.zeros(xs_a.shape[1] + 1)
                else:
                    # Find line
                    beta, dist_x = \
                        lowessNd(self.theta,
                                 xs_a,
                                 ts_a[:, [j]],
                                 self.kernel_lowess)
                    beta_p, dist_x_p = \
                        lowessNd(self.theta_p,
                                 xs_a,
                                 ts_a[:, [j]],
                                 self.kernel_lowess)

                # Translate and shear
                x_t = xs_a - self.theta
                x_t_p = xs_a - self.theta_p

                sheared_y[j] = ts_a[:, [j]] - \
                    np.sum(beta[1:] * x_t, axis=1, keepdims=True)
                sheared_y_p[j] = ts_a[:, [j]] - \
                    np.sum(beta_p[1:] * x_t_p, axis=1, keepdims=True)

            alphas = np.zeros(self.M)
            dim = xs_a.shape[1]

            for m in xrange(self.M):
                # Get bootstrap sample using the weights
                bs_ids = get_bootstrap_ids(self.current_sim_calls,
                                           weights_x)
                bs_ids_p = get_bootstrap_ids(self.current_sim_calls,
                                             weights_x_p)

                if self.smoothed_bs:
                    num_bs = bs_ids.shape[0]
                    num_bs_p = bs_ids_p.shape[0]

                    # Use a smoothed bootstrap
                    noise = np.random.multivariate_normal(
                                np.zeros(dim),
                                np.diag(np.repeat(1.0 / np.sqrt(swx), dim)),
                                num_bs)
                    noise_p = np.random.multivariate_normal(
                                np.zeros(dim),
                                np.diag(np.repeat(1.0 / np.sqrt(swx_p), dim)),
                                num_bs_p)
                else:
                    noise = 0
                    noise_p = 0

                other_term = 0.0

                # For each y-dimension perform a KDE.
                for j in xrange(self.y_dim):
                    other_term += \
                        km.kernel_density_estimate(
                            self.y_star[j],
                            sheared_y_p[j][bs_ids_p] + noise_p,
                            self.kernel_y,
                            self.h_y_method) - \
                        km.kernel_density_estimate(
                            self.y_star[j],
                            sheared_y[j][bs_ids] + noise,
                            self.kernel_y,
                            self.h_y_method)

                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate / grid values
            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error > self.ksi:
                # Acquire training points
                self.get_samples()
            else:
                return distr.uniform.rvs() < tau
