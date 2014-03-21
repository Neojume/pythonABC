import kernels
import sys
import numpy as np
import distributions as distr
from numpy import linalg
import data_manipulation as dm


def logsumexp(x, dim=0):
    '''
    Compute log(sum(exp(x))) in numerically stable way.
    '''

    if dim == 0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x - xmax).sum(0))
    # TODO: support other dimensions
    # NOTE: Newaxis is not defined
    #elif dim == 1:
    #    xmax = x.max(1)
    #    return xmax + np.log(np.exp(x - xmax[:, newaxis]).sum(1))
    else:
        raise 'dim ' + str(dim) + 'not supported'


def pseudo_marginal_ABC(problem, num_samples, epsilon, S, verbose=False,
                        save=True):
    '''
    Performs the Pseudo-Marginal Likelihood ABC algorithm described by Meeds
    and Welling.

    Parameters
    ----------
    problem : An instance of (a subclass of) ABC_Problem.
        The problem to solve.
    epsilon : float
        Error margin.
    S : int
        Number of simulations per iteration.
    num_samples : int
        The number of samples to draw.
    verbose : bool
        The verbosity of the algorithm. If True, will print iteration numbers
        and number of simulation calls
    save : bool
        If True, results will be stored in a possibly existing database

    Returns
    -------
    samples, sim_calls, accepted : tuple
        samples: list of samples

        sim_calls: list of simulation calls needed for each sample

        accepted: list of bools whether the sample was accepted for each sample
    '''

    # Make local copies of problem parameters for speed

    y_star = problem.y_star

    prior = problem.prior
    prior_args = problem.prior_args

    proposal = problem.proposal
    proposal_args = problem.proposal_args
    use_log = problem.use_log

    simulator = problem.simulator

    theta = problem.theta_init
    if use_log:
        log_theta = np.log(theta)

    prev_diff = []

    for s in xrange(S):
        new_x = simulator(theta)

        # Compute the P(y | x, theta) for these samples
        u = linalg.norm(new_x - y_star) / epsilon
        prev_diff.append(kernels.log_gaussian(u / epsilon))

    prev_diff_term = logsumexp(np.array(prev_diff))
    cur_sim_calls = S

    samples = []
    accepted = []
    sim_calls = []

    for i in xrange(num_samples):
        if verbose:
            if i % 100 == 0:
                sys.stdout.write('\riteration %d %d' % (i, sum(sim_calls)))
                sys.stdout.flush()

        # Propose a new theta
        if use_log:
            theta_p = proposal.rvs(log_theta, *proposal_args)
            log_theta_p = np.log(theta_p)
        else:
            theta_p = proposal.rvs(theta, *proposal_args)

        diff_p = []

        # Get S samples and approximate marginal likelihood
        for s in xrange(S):
            new_x_p = simulator(theta_p)

            # Compute the P(y | x, theta) for these samples
            u_p = linalg.norm(new_x_p - y_star) / epsilon
            diff_p.append(kernels.log_gaussian(u_p / epsilon))

        cur_sim_calls += S

        # Calculate acceptance according to eq. 4
        numer = prior.logpdf(theta_p, *prior_args)
        denom = prior.logpdf(theta, *prior_args)

        if use_log:
            numer += proposal.logpdf(theta, log_theta_p, *proposal_args)
            denom += proposal.logpdf(theta_p, log_theta, *proposal_args)
        else:
            numer += proposal.logpdf(theta, theta_p, *proposal_args)
            denom += proposal.logpdf(theta_p, theta, *proposal_args)

        cur_diff_term = logsumexp(np.array(diff_p))
        diff_term = cur_diff_term - prev_diff_term

        log_alpha = min(0.0, (numer - denom) + diff_term)

        # Accept proposal with probability alpha
        if distr.uniform.rvs(0, 1) <= np.exp(log_alpha):
            accepted.append(True)
            theta = theta_p
            if use_log:
                log_theta = log_theta_p
            prev_diff_term = cur_diff_term
        else:
            accepted.append(False)

        samples.append(theta)
        sim_calls.append(cur_sim_calls)
        cur_sim_calls = 0

    if verbose:
        print ''

    if save:
        dm.save(pseudo_marginal_ABC, [epsilon, S], problem,
                (samples, sim_calls, accepted))

    return samples, sim_calls, accepted


def marginal_ABC(problem, num_samples, epsilon, S, verbose=False, save=True):
    '''
    Performs the Marginal Likelihood ABC algorithm described by Meeds
    and Welling.

    Parameters
    ----------
    problem : An instance of (a subclass of) ABC_Problem.
        The problem to solve.
    S : int
        Number of simulations per iteration.
    epsilon : float
        Error margin.
    num_samples : int
        The number of samples to draw.
    verbose : bool
        The verbosity of the algorithm. If True, will print iteration numbers
        and number of simulation calls
    save : bool
        If True, results will be stored in a possibly existing database

    Returns
    -------
    samples, sim_calls, accepted : tuple
        samples: list of samples

        sim_calls: list of simulation calls needed for each sample

        accepted: list of bools whether the sample was accepted for each sample
    '''

    # Make local copies of problem parameters for speed

    y_star = problem.y_star

    prior = problem.prior
    prior_args = problem.prior_args

    proposal = problem.proposal
    proposal_args = problem.proposal_args
    use_log = problem.use_log

    simulator = problem.simulator

    theta = problem.theta_init
    log_theta = np.log(theta)

    samples = []
    accepted = []
    sim_calls = []

    for i in xrange(num_samples):
        if verbose:
            if i % 100 == 0:
                sys.stdout.write('\riteration %d %d' % (i, sum(sim_calls)))
                sys.stdout.flush()

        # Propose a new theta
        if use_log:
            theta_p = proposal.rvs(log_theta, *proposal_args)
            log_theta_p = np.log(theta_p)
        else:
            theta_p = proposal.rvs(theta, *proposal_args)

        diff = []
        diff_p = []

        current_sim_calls = 0

        # Get S samples and approximate marginal likelihood
        for s in xrange(S):
            new_x = simulator(theta)
            new_x_p = simulator(theta_p)
            current_sim_calls += 2

            # Compute the P(y | x, theta) for these samples
            u = linalg.norm(new_x - y_star) / epsilon
            u_p = linalg.norm(new_x_p - y_star) / epsilon

            diff.append(kernels.log_gaussian(u / epsilon))
            diff_p.append(kernels.log_gaussian(u_p / epsilon))

        # Calculate acceptance according to eq. 4
        numer = prior.logpdf(theta_p, *prior_args)
        denom = prior.logpdf(theta, *prior_args)

        if use_log:
            numer += proposal.logpdf(theta, log_theta_p, *proposal_args)
            denom += proposal.logpdf(theta_p, log_theta, *proposal_args)
        else:
            numer += proposal.logpdf(theta, theta_p, *proposal_args)
            denom += proposal.logpdf(theta_p, theta, *proposal_args)

        diff_term = logsumexp(np.array(diff_p)) - logsumexp(np.array(diff))

        log_alpha = min(0.0, (numer - denom) + diff_term)

        # Accept proposal with probability alpha
        if distr.uniform.rvs(0, 1) <= np.exp(log_alpha):
            accepted.append(True)
            theta = theta_p
            if use_log:
                log_theta = log_theta_p
        else:
            accepted.append(False)

        samples.append(theta)
        sim_calls.append(current_sim_calls)

    if verbose:
        print ''

    if save:
        dm.save(marginal_ABC, [epsilon, S], problem,
                (samples, sim_calls, accepted))

    return samples, sim_calls, accepted
