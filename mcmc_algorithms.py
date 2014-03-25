import sys
import numpy as np
from numpy import linalg
import kernels
from abc import ABCMeta, abstractmethod
import distributions as distr
import data_manipulation as dm
from utils import logsumexp, conditional_error


class Base_MCMC_ABC(object):

    '''
    Abstract base class for MCMC ABC.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, problem, num_samples, verbose=False, save=True,
                 **kwargs):
        self.y_star = problem.y_star
        self.y_dim = problem.y_dim

        self.simulator = problem.simulator

        self.prior = problem.prior
        self.prior_args = problem.prior_args

        self.proposal = problem.proposal
        self.proposal_args = problem.proposal_args
        self.use_log = problem.use_log

        self.theta = problem.theta_init
        if self.use_log:
            self.log_theta = np.log(self.theta)

        self.num_samples = num_samples
        self.samples = []
        self.sim_calls = []
        self.accepted = []
        self.current_sim_calls = 0

        self.verbose = verbose
        self.save = save

    @abstractmethod
    def save(self):
        return NotImplemented

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
        for i in xrange(self.num_samples):
            if self.verbose and i % 200 == 0:
                sys.stdout.write('\r%s iteration %d %d' %
                                 (type(self).__name__, i, sum(self.sim_calls)))
                sys.stdout.flush()

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
            self.save()


class marginal_ABC(Base_MCMC_ABC):

    def __init__(self, problem, num_samples, **params):
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
        num_samples : int
            The number of samples to draw.
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

        super(marginal_ABC, self).__init__(problem, num_samples, **params)

        assert set(['S', 'epsilon']).issubset(params.keys()), \
            'Not enough parameters: need S and epsilon'

        self.S = params['S']
        self.epsilon = params['epsilon']

    def save(self):
        dm.save(marginal_ABC, [self.epsilon, self.S], self.problem,
                (self.samples, self.sim_calls, self.accepted))

    def mh_step(self):
        diff = []
        diff_p = []

        # Get S samples and approximate marginal likelihood
        for s in xrange(self.S):
            new_x = self.simulator(self.theta)
            new_x_p = self.simulator(self.theta_p)
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


class pseudo_marginal_ABC(Base_MCMC_ABC):

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
        super(pseudo_marginal_ABC, self).__init__(problem, num_samples,
                                                  **params)

        assert set(['S', 'epsilon']).issubset(params.keys()), \
            'Not enough parameters: need S and epsilon'

        self.S = params['S']
        self.epsilon = params['epsilon']

        prev_diff = []

        for s in xrange(self.S):
            new_x = self.simulator(self.theta)

            # Compute the P(y | x, theta) for these samples
            u = linalg.norm(new_x - self.y_star) / self.epsilon
            prev_diff.append(kernels.log_gaussian(u / self.epsilon))

        self.prev_diff_term = logsumexp(np.array(prev_diff))
        self.cur_sim_calls = self.S

    def save(self):
        dm.save(pseudo_marginal_ABC, [self.epsilon, self.S], self.problem,
                (self.samples, self.sim_calls, self.accepted))

    def mh_step(self):
        diff_p = []

        # Get S samples and approximate marginal likelihood
        for s in xrange(self.S):
            new_x_p = self.simulator(self.theta_p)

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


class SL_ABC(Base_MCMC_ABC):

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

        self.eye = np.identity(self.y_dim)

        assert set(['S', 'epsilon']).issubset(params.keys()), \
            'Not enough parameters: need S and epsilon'

        self.S = params['S']
        self.epsilon = params['epsilon']

    def save(self):
        dm.save(SL_ABC, [self.epsilon, self.S], self.problem,
                (self.samples, self.sim_calls, self.accepted))

    def mh_step(self):
        # Get S samples from simulator
        x = [self.simulator(self.theta) for s in xrange(self.S)]
        x_p = [self.simulator(self.theta_p) for s in xrange(self.S)]
        self.current_sim_calls = 2 * self.S

        # Set mu's according to eq. 5
        mu_theta = np.mean(x)
        mu_theta_p = np.mean(x_p)

        # Set sigma's according to eq. 6
        sigma_theta = np.std(x)
        sigma_theta_p = np.std(x_p)

        # Calculate acceptance according to eq. 10
        numer = self.prior_logprob_p + self.proposal_logprob
        denom = self.prior_logprob + self.proposal_logprob_p

        other_term = distr.normal.logpdf(
            self.y_star,
            mu_theta_p,
            sigma_theta_p + (self.epsilon ** 2) * self.eye) - \
            distr.normal.logpdf(
                self.y_star,
                mu_theta,
                sigma_theta + (self.epsilon ** 2) * self.eye)

        log_alpha = min(0.0, (numer - denom) + other_term)

        return distr.uniform.rvs(0.0, 1.0) <= np.exp(log_alpha)


class ASL_ABC(Base_MCMC_ABC):

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
        verbose : bool
            The verbosity of the algorithm. If `True`, will print iteration
            numbers and number of simulations. Default `False`.
        save : bool
            If `True`, results will be stored in a possibly existing database
            Default `True`.

        Note that while epsilon, ksi, S0 and delta_S are keyword-arguments,
        they are necessary.

        Optional Parameters
        -------------------
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

        needed_params = ['S0', 'epsilon', 'ksi', 'delta_S']
        assert set(needed_params).issubset(params.keys()), \
            'Not enough parameters: Need {0}'.format(str(needed_params))

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

    def save(self):
        dm.save(ASL_ABC, [self.epsilon, self.ksi, self.S0, self.delta_S],
                self.problem, (self.samples, self.sim_calls, self.accepted))

    def mh_step(self):
        # Reset the samples
        x = []
        x_p = []

        additional = self.S0

        while True:
            # Get additional samples from simulator
            x.extend([self.simulator(self.theta) for s in xrange(additional)])
            x_p.extend([self.simulator(self.theta_p) for s in xrange(additional)])

            additional = self.delta_S
            S = len(x)

            # Set mu's according to eq. 5
            mu_hat_theta = np.mean(x)
            mu_hat_theta_p = np.mean(x_p)

            # Set sigma's according to eq. 6
            # TODO: incremental implementation?
            sigma_theta = np.std(x)
            sigma_theta_p = np.std(x_p)

            sigma_theta_S = sigma_theta / float(S)
            sigma_theta_p_S = sigma_theta_p / float(S)

            alphas = []
            for m in range(self.M):
                # Sample mu_theta_p and mu_theta using eq. 11
                mu_theta = distr.normal.rvs(mu_hat_theta, sigma_theta_S)
                mu_theta_p = distr.normal.rvs(mu_hat_theta_p, sigma_theta_p_S)

                # Compute alpha using eq. 12
                numer = self.prior_logprob_p + self.proposal_logprob + \
                    distr.normal.logpdf(
                        self.y_star,
                        mu_theta_p,
                        sigma_theta_p + self.eye_eps)

                denom = self.prior_logprob + self.proposal_logprob_p + \
                    distr.normal.logpdf(
                        self.y_star,
                        mu_theta,
                        sigma_theta + self.eye_eps)

                log_alpha = min(0.0, numer - denom)
                alphas.append(np.exp(log_alpha))

            alphas = np.array(alphas)
            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate
            error = np.mean([e * conditional_error(alphas, e, tau, self.M)
                            for e in np.linspace(0, 1, self.E)])

            if error < self.ksi:
                break

        self.current_sim_calls = 2 * S

        return distr.uniform.rvs(0.0, 1.0) <= tau


if __name__ == '__main__':
    from problems import wilkinson_problem
    import matplotlib.pyplot as plt

    problem = wilkinson_problem()

    alg = ASL_ABC(problem, 10000, epsilon=0.05, S0=5, delta_S=5, ksi=0.05,
                  save=False, verbose=True)
    alg.run()

    print sum(alg.accepted) / float(alg.num_samples)
    plt.hist(np.array(alg.samples), 100)
    plt.show()
