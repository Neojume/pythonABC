# -*- coding: utf-8 -*-
"""
KRS-ABC

Kernel Regressian Surrogate - Approximate Bayesian Computation

@author: steven
"""

import numpy as np
from random import choice
import distributions as distr
from kernel_regression import kernel_regression
import data_manipulation as dm

import matplotlib.pyplot as plt


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
    #plt.ylim(-4, 14)
    plt.title('S = {0}, h = {1}'.format(len(xs), h))


def conditional_error(alphas, u, tau, M):
    if u <= tau:
        return float(np.sum(alphas < u)) / M
    else:
        return float(np.sum(alphas >= u)) / M


def KRS_ABC(problem, num_samples, epsilon, ksi, S0, delta_S, verbose=False,
            save=True):

    M = 50
    E = 50

    y_star = problem.y_star
    eye = np.identity(problem.y_dim)

    simulator = problem.simulator

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

    xs = list(prior.rvs(*prior_args, N=S0))
    ts = [simulator(x) for x in xs]

    current_sim_calls = S0
    sim_calls = []
    accepted = []

    h = np.var(xs) * pow(3.0 / (4 * len(xs)), 0.2) / 8.0
    h = 0.1

    rng = np.linspace(problem.rng[0], problem.rng[1])

    for i in xrange(num_samples):
        if verbose:
            if i % 200 == 0:
                print i
                #plot_krs(xs, ts, h, rng, y_star)
                # plt.show()

        # Sample theta_p from proposal
        theta_p = proposal.rvs(theta, *proposal_args)
        if use_log:
            log_theta_p = np.log(theta_p)

        while True:
            # Get mus and sigmas from Kernel regression
            mu_bar, std, conf, N = kernel_regression(theta, xs, ts, h)
            mu_bar_p, std_p, conf_p, N_p = kernel_regression(
                theta_p, xs, ts, h)

            # Get samples
            mu = distr.normal.rvs(mu_bar, std / np.sqrt(N), M)
            mu_p = distr.normal.rvs(mu_bar_p, std_p / np.sqrt(N_p), M)

            # Compute alpha using eq. 19
            numer = prior.logpdf(theta_p, *prior_args)
            denom = prior.logpdf(theta, *prior_args)

            if use_log:
                numer += proposal.logpdf(theta, log_theta_p, *proposal_args)
                denom += proposal.logpdf(theta_p, log_theta, *proposal_args)
            else:
                numer += proposal.logpdf(theta, theta_p, *proposal_args)
                denom += proposal.logpdf(theta_p, theta, *proposal_args)

            alphas = np.zeros(M)
            for m in xrange(M):
                other_term = \
                    distr.normal.logpdf(
                        y_star,
                        mu_p[m],
                        std_p + (epsilon ** 2)) - \
                    distr.normal.logpdf(
                        y_star,
                        mu[m],
                        std + (epsilon ** 2))

                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            error = np.mean([e * conditional_error(alphas, e, tau, M)
                            for e in np.linspace(0, 1, E)])

            if error > ksi:
                # Acquire training point

                new_x = problem.prior.rvs(*prior_args)
                #new_x = choice([theta, theta_p])
                ts.append(simulator(new_x))
                xs.append(new_x)

                #h = np.var(xs) * pow(3.0 / (4 * len(xs)), 0.2) / 8.0

                current_sim_calls += 1
                # if sim_calls % 1000 == 0:
                #    print error
                #    plot_krs(xs, ts, h, rng, y_star)
                #    plt.axvline(theta)
                #    plt.axvline(theta_p)
                #    plt.show()
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
    from problems import *
    from compare import variation_distance

    problem = wilkinson_problem()
    for i in range(4):
        samples, sim_calls, accepted = KRS_ABC(
            problem, 10000, 0.05, 0.1, 100, 10, True)

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
