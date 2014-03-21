import numpy as np
import distributions as distr
import data_manipulation as dm
import sys


def conditional_error(alphas, u, tau, M):
    if u <= tau:
        return float(np.sum(alphas < u)) / M
    else:
        return float(np.sum(alphas >= u)) / M


def ASL_ABC(problem, num_samples, epsilon, ksi, S0, delta_S, verbose=False,
            save=True):
    '''
    Performs the Adaptive Synthetic Likelihood ABC algorithm described by Meeds
    and Welling.

    Parameters
    ----------
    problem : An instance of (a subclass of) ABC_Problem.
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
        The verbosity of the algorithm. If True, will print iteration
        numbers and number of simulation calls
    save : bool
        If True, results will be stored in a possibly existing database

    Returns
    -------
    samples, sim_calls, accepted : tuple
        samples: list of samples

        sim_calls: list of simulation calls needed for each sample

        accepted: list of bools whether the sample was accepted for each sample
    '''

    y_star = problem.y_star
    eye = np.identity(problem.y_dim)

    # Prior distribution
    prior = problem.prior
    prior_args = problem.prior_args

    # Proposal distribution
    # NOTE: First arg is always theta.
    proposal = problem.proposal
    proposal_args = problem.proposal_args
    use_log = problem.use_log

    simulator = problem.simulator

    theta = problem.theta_init
    if use_log:
        log_theta = np.log(theta)

    samples = []
    sim_calls = []
    accepted = []

    for i in range(num_samples):
        # Sample theta_p from proposal
        if use_log:
            theta_p = proposal.rvs(log_theta, *proposal_args)
            log_theta_p = np.log(theta_p)
        else:
            theta_p = proposal.rvs(theta, *proposal_args)

        # Calculate constant probs wrt mu_hat outside loop
        prior_logprob_p = prior.logpdf(theta_p, *prior_args)
        prior_logprob = prior.logpdf(theta, *prior_args)

        if use_log:
            proposal_logprob = proposal.logpdf(
                theta, log_theta_p, *proposal_args)
            proposal_logprob_p = proposal.logpdf(
                theta_p, log_theta, *proposal_args)
        else:
            proposal_logprob = proposal.logpdf(theta, theta_p, *proposal_args)
            proposal_logprob_p = proposal.logpdf(
                theta_p, theta, *proposal_args)

        # Reset the samples
        x = []
        x_p = []

        additional = S0

        M = 50

        while True:
            # Get additional samples from simulator
            x.extend([simulator(theta) for s in xrange(additional)])
            x_p.extend([simulator(theta_p) for s in xrange(additional)])

            additional = delta_S
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
            for m in range(M):
                # Sample mu_theta_p and mu_theta using eq. 11
                mu_theta = distr.normal.rvs(mu_hat_theta, sigma_theta_S)
                mu_theta_p = distr.normal.rvs(mu_hat_theta_p, sigma_theta_p_S)

                # Compute alpha using eq. 12
                numer = prior_logprob_p + proposal_logprob + \
                    distr.normal.logpdf(
                        y_star,
                        mu_theta_p,
                        sigma_theta_p + (epsilon ** 2) * eye)

                denom = prior_logprob + proposal_logprob_p + \
                    distr.normal.logpdf(
                        y_star,
                        mu_theta,
                        sigma_theta + (epsilon ** 2) * eye)

                log_alpha = min(0.0, numer - denom)
                alphas.append(np.exp(log_alpha))

            alphas = np.array(alphas)
            tau = np.median(alphas)

            # Set unconditional error, using Monte Carlo estimate
            E = 50
            error = np.mean([e * conditional_error(alphas, e, tau, M)
                            for e in np.linspace(0, 1, E)])

            if error < ksi:
                break

        current_sim_calls = 2 * S

        if distr.uniform.rvs(0.0, 1.0) <= tau:
            accepted.append(True)
            theta = theta_p
            if use_log:
                log_theta = np.log(theta_p)
        else:
            accepted.append(False)

        # Add the sample to the set of samples
        samples.append(theta)
        sim_calls.append(current_sim_calls)

        if verbose:
            if i % 200 == 0:
                sys.stdout.write('\riteration %d %d' % (i, sum(sim_calls)))
                sys.stdout.flush()

    if verbose:
        print ''

    if save:
        dm.save(ASL_ABC, [epsilon, ksi, S0, delta_S], problem,
                (samples, sim_calls, accepted))

    return samples, sim_calls, accepted
