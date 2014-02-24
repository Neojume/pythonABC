# -*- coding: utf-8 -*-
"""
KRS-ABC

Kernel Regressian Surrogate - Approximate Bayesian Computation

@author: steven
"""

import numpy as np
import distributions as distr
from kernel_regression import kernel_regression

import matplotlib.pyplot as plt

def plot_krs(xs, ts, h, rng, y_star):
    means = np.zeros(50)
    stds = np.zeros(50)
    confs = np.zeros(50)
    for i, val in enumerate(rng):
        means[i], stds[i], confs[i], _ = kernel_regression(val, xs, ts, h)

    
    plt.fill_between(
            rng,
            means - confs, 
            means + confs, 
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
    plt.ylim(-2,4)
    plt.title('S = {0}, h = {1}'.format(len(xs), h))
    plt.show()

def conditional_error(alphas, u, tau, M):
    if u <= tau:
        return float(np.sum(alphas < u)) / M
    else:
        return float(np.sum(alphas >= u)) / M

def KRS_ABC(problem, num_samples, epsilon, ksi, S0, delta_S):
    
    M = 50
    E = 50    
    
    eye = np.identity(1)

    y_star = problem.y_star

    simulator = problem.simulator

    # Prior distribution
    prior = problem.prior 
    prior_args = problem.prior_args
    
    # Proposal distribution
    # NOTE: First arg is always theta.
    proposal = problem.proposal
    proposal_args = problem.proposal_args
                  
    theta = problem.theta_init
    log_theta = np.log(theta)
    
    samples = np.zeros(num_samples)
    
    xs = list(prior.rvs(*prior_args, N=S0))
    ts = [simulator(x)[0] for x in xs]
    
    sim_calls = 0
    accepted = 0
    
    h = np.var(xs) * pow(3.0 / (4 * len(xs)), 0.2) / 8.0

    rng = np.linspace(-np.pi, 5 * np.pi)

    for i in xrange(num_samples):
        if i % 100 == 0:
            print i, sim_calls
            #plot_krs(xs, ts, h, rng, y_star)
            
        # Sample theta_p from proposal
        theta_p = prior.rvs(*prior_args)#proposal.rvs(theta, *proposal_args)
        #log_theta_p = np.log(theta_p)
        
        while True:
            # Get mus and sigmas from Kernel regression
            mu_bar, sigma_j, sigma, N = kernel_regression(theta, xs, ts, h)
            mu_bar_p, sigma_j_p, sigma_p, N_p = kernel_regression(theta_p, xs, ts, h)
           
            # Get samples
            mu = distr.normal.rvs(mu_bar, sigma / N, M)
            mu_p = distr.normal.rvs(mu_bar_p, sigma_p / N_p, M)
            
            # Compute alpha using eq. 19
            numer = prior.logpdf(theta_p, *prior_args) + \
                    proposal.logpdf(theta, theta_p, *proposal_args)
            denom = prior.logpdf(theta, *prior_args) + \
                    proposal.logpdf(theta_p, theta, *proposal_args)

            alphas = np.zeros(M)
            for m in xrange(M):
                other_term = \
                    distr.normal.logpdf(y_star, mu_p[m], sigma_j_p + (epsilon ** 2)) - \
                    distr.normal.logpdf(y_star, mu[m], sigma_j + (epsilon ** 2))        
            
                log_alpha = min(0.0, (numer - denom) + other_term)
                alphas[m] = np.exp(log_alpha)

            tau = np.median(alphas)

            error = np.mean([e * conditional_error(alphas, e, tau, M) \
                    for e in np.linspace(0, 1, E)])        

            #print alphas
            #print tau
            #print error
            #raw_input()
            
            if error > ksi:
                # Acquire training point
                
                # For now we add training points at theta and theta_p
                new_x = prior.rvs(*prior_args)
                ts.append(simulator(new_x)[0])
                xs.append(new_x)
                
                h = np.var(xs) * pow(3.0 / (4 * len(xs)), 0.2) / 8.0

                sim_calls += 1
                #if sim_calls % 300 == 0:
                #    plot_krs(xs, ts, h, rng, y_star)
            else:
                break
            
        if distr.uniform.rvs() <= tau:
            #print 'ACCEPTEEEEEEEEEEEEEEEEEEEEEEEED', i

            accepted += 1
            theta = theta_p
            #log_theta = log_theta_p
        else:
            #print 'REJECTEEEEEEEEEEEEEEEEEEEEEEEED', i
            pass

        samples[i] = theta
    
    return samples, float(accepted) / num_samples, sim_calls

if __name__ == '__main__':
    from problems import toy_problem, sinus_problem
    problem = sinus_problem()
    samples, rate, sim_calls = KRS_ABC(problem, 10000, 0.1, 0.05, 20, 10)
    
    print sim_calls, rate   
    
    #rng = np.linspace(0.07, 0.13)
    plt.hist(samples[1500:], bins = 100, normed=True)
    #plt.plot(rng, np.exp(distr.gamma.logpdf(rng, 0.1 + 500, (0.1 + 500*9.42))))

    plt.show()
