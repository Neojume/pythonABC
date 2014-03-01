import numpy as np
import distributions as distr
import matplotlib.pyplot as plt


def SL_ABC(problem, num_samples, epsilon, S, verbose=False):
    '''
    Performs the Synthetic Likelihood ABC algorithm described by Meeds and
    Welling.

    Parameters
    ----------
    problem : An instance of (a subclass of) ABC_Problem.
        The problem to solve.
    S : int
        Number of simulations per iteration
    epsilon : float
        Error margin.
    num_samples : int
        The number of samples
    verbose : bool
        The verbosity of the algorithm. If True, will print iteration
        numbers and number of simulations

    Returns
    -------
    samples, sim_calls, rate : tuple
        samples: list of samples
        sim_calls: list of simulation calls needed for each sample
        rate: the acceptance rate
    '''

    # Make local copies of problem parameters for speed
    y_star = problem.y_star
    eye = np.identity(problem.y_dim)

    prior = problem.prior
    prior_args = problem.prior_args

    proposal = problem.proposal
    proposal_args = problem.proposal_args

    simulator = problem.simulator

    samples = []
    sim_calls = []
    accepted = 0

    theta = problem.theta_init
    log_theta = np.log(theta)

    for i in xrange(num_samples):
        # Sample theta_p from proposal
        theta_p = proposal.rvs(log_theta, *proposal_args)
        log_theta_p = np.log(theta_p)

        # Get S samples from simulator
        x = [simulator(theta) for s in xrange(S)]
        x_p = [simulator(theta_p) for s in xrange(S)]
        current_sim_calls = 2 * S

        # Set mu's according to eq. 5
        mu_theta = np.mean(x)
        mu_theta_p = np.mean(x_p)

        # Set sigma's according to eq. 6
        sigma_theta = np.std(x)
        sigma_theta_p = np.std(x_p)

        # Compute alpha using eq. 10
        numer = prior.logpdf(theta_p, *prior_args) + \
            proposal.logpdf(theta, log_theta_p, *proposal_args)
        denom = prior.logpdf(theta, *prior_args) + \
            proposal.logpdf(theta_p, log_theta, *proposal_args)

        other_term = distr.normal.logpdf(y_star,
                mu_theta_p,
                sigma_theta_p + (epsilon ** 2) * eye) - \
            distr.normal.logpdf(y_star,
                mu_theta,
                sigma_theta + (epsilon ** 2) * eye)

        log_alpha = min(0.0, (numer - denom) + other_term)

        if distr.uniform.rvs(0.0, 1.0) <= np.exp(log_alpha):
            accepted += 1
            log_theta = log_theta_p
            theta = theta_p

        # Accept the sample
        samples.append(theta)
        sim_calls.append(current_sim_calls)

        if verbose:
            if i % 200 == 0:
                sys.stdout.write('\riteration %d %d' % (i, sum(sim_calls)))
                sys.stdout.flush()

    if verbose:
        print ''

    return samples, sim_calls, float(accepted) / num_samples

if __name__ == '__main__':
    from problems import toy_problem

    problem = toy_problem()

    samples, sim_calls, acceptance_rate = SL_ABC(problem, 10000, 0, 10, True)

    print 'sim_calls', sum(sim_calls)
    print 'acceptance rate', acceptance_rate

    post = problem.true_posterior
    post_args = problem.true_posterior_args

    precision = 100
    test_range = np.linspace(0.07, 0.13, 100)
    plt.plot(test_range, np.exp(post.logdf(test_range, *post_args)))
    plt.hist(samples[1500:], 100, normed=True, alpha=0.5)
    plt.show()
