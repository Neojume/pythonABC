import kernels
import numpy as np
import distributions as distr
from numpy import linalg

import matplotlib.pyplot as plt


def logsumexp(x, dim=0):
    '''
    Compute log(sum(exp(x))) in numerically stable way.
    '''

    if dim == 0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x - xmax).sum(0))
    elif dim == 1:
        xmax = x.max(1)
        return xmax + np.log(np.exp(x - xmax[:, newaxis]).sum(1))
    else:
        raise 'dim ' + str(dim) + 'not supported'


def marginal_ABC(problem, num_samples, epsilon, S, verbose=False):
    '''
    Performs the Pseudo Marginal Likelihood ABC algorithm described by Meeds
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
    '''

    # Make local copies of problem parameters for speed

    y_star = problem.y_star

    prior = problem.prior
    prior_args = problem.prior_args

    proposal = problem.proposal
    proposal_args = problem.proposal_args

    simulator = problem.simulator

    theta = problem.theta_init
    log_theta = np.log(theta)

    samples = []
    sim_calls = []
    accepted = 0

    for i in xrange(num_samples):
        if verbose:
            if i % 100 == 0:
                print 'iteration', i, sum(sim_calls)
        # Propose a new theta
        theta_p = proposal.rvs(log_theta, *proposal_args)
        log_theta_p = np.log(theta_p)

        x = []
        x_p = []

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

            x.append(new_x)
            x_p.append(new_x_p)

        numer = prior.logpdf(theta_p, *prior_args) + \
            proposal.logpdf(theta, log_theta_p, *proposal_args)
        denom = prior.logpdf(theta, *prior_args) + \
            proposal.logpdf(theta_p, log_theta, *proposal_args)

        diff_term = logsumexp(np.array(diff_p)) - logsumexp(np.array(diff))

        log_alpha = min(0.0, (numer - denom) + diff_term)

        if distr.uniform.rvs(0, 1) <= np.exp(log_alpha):
            accepted += 1
            theta = theta_p
            log_theta = log_theta_p

        samples.append(theta)
        sim_calls.append(current_sim_calls)

    return samples, sim_calls, accepted / float(num_samples)

if __name__ == '__main__':
    from problems import toy_problem

    problem = toy_problem()
    samples, sim_calls, rate = marginal_ABC(problem, 10000, 0.05, 20)

    print 'sim_calls', sum(sim_calls)
    print 'acceptance ratio', rate

    post = problem.true_posterior
    post_args = problem.true_posterior_args

    # Create plots of how close we are
    rng = np.linspace(0.07, 0.13, 100)
    plt.hist(samples[1500:], bins=100, normed=True)
    plt.plot(rng, np.exp(post.logpdf(rng, *post_args)))
    plt.show()
