# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:44:33 2014

@author: steven
"""
import cProfile
import numpy as np
from random import uniform, gauss, expovariate

import distributions as distr

import scipy.stats as stats
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import seed

def gaussian(y, x, epsilon):
    J = len(y)
    return np.exp( - (x - y).T * (x - y) / (2 * epsilon ** 2)) / np.power(2 * np.pi * epsilon, J / 2)  
        
def conditional_error(alphas, u, tau, M):
    if u <= tau:
        return float(np.sum(alphas < u)) / M
    else:
        return float(np.sum(alphas >= u)) / M
        
def simulator(theta, N=500):
    return np.mean(distr.exponential.rvs(theta, N))

    

#if __name__ == '__main__':
def ASL_ABC():
    # So that the result is always the same
    #np.random.seed(87655678)
    #seed(87655678)
    
    # Parameters
    num_samples = 10000
    y_star = 9.42
    epsilon = 0.0
    ksi = 0.01
    S0 = 5
    delta_S = 10

    eye = np.identity(1)

    # Prior distribution
    prior = distr.gamma
    prior_args = [0.1, 0.1]
    
    # Proposal distribution
    # NOTE: First arg is always theta.
    proposal = distr.lognormal
    proposal_args = [0.1]    
                  
    log_theta = 0.0
    theta = 1.0
    
    samples = []    
    sim_calls = 0
            
    for i in range(num_samples):
        if i % 200 == 0:
            print i

        # Sample theta_p from proposal
        theta_p = proposal.rvs(log_theta, *proposal_args)
        log_theta_p = np.log(theta_p)

        # Calculate constant probs wrt mu_hat outside loop 
        prior_logprob_p = prior.logpdf(theta_p, *prior_args)
        prior_logprob = prior.logpdf(theta, *prior_args)
        
        proposal_logprob = proposal.logpdf(theta, log_theta_p, *proposal_args)
        proposal_logprob_p = proposal.logpdf(theta_p, log_theta, *proposal_args)
        
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
                mu_theta_p = distr.normal.rvs(mu_hat_theta_p, sigma_theta_p_S)
                mu_theta = distr.normal.rvs(mu_hat_theta, sigma_theta_S)
                
                # Compute alpha using eq. 12   
                numer = prior_logprob_p + distr.normal.logpdf(y_star, mu_theta_p, sigma_theta_p + (epsilon ** 2) * eye) + proposal_logprob
                denom = prior_logprob + distr.normal.logpdf(y_star, mu_theta, sigma_theta + (epsilon ** 2) * eye) + proposal_logprob_p
                
                log_alpha = min(0.0, numer - denom)
                alphas.append(np.exp(log_alpha))

            alphas = np.array(alphas)
            tau = np.median(alphas)
            
            # Set unconditional error, using Monte Carlo estimate
            E = 50
            #cond_errors = [conditional_error(alphas, uniform(0.0, 1.0), tau, M) for e in xrange(E)]
            #error = sum([i * conditional_error(alphas, i, tau, M) for i in np.linspace(0.0, 1.0, E)])
            #error = np.mean(cond_errors)
            error = np.mean([i * conditional_error(alphas, i, tau, M) for i in np.linspace(0,1,E)])
            #error = integrate.quad(lambda x : x * conditional_error(alphas, x, tau, M), 0.0, 1.0)

            if error < ksi:
                break
        
        sim_calls += S
        if uniform(0.0, 1.0) <= tau:

            theta = theta_p
            log_theta = np.log(theta_p)
        
        # Accept the sample
        samples.append(theta)

    print sim_calls
    
    #test_range = np.linspace(0, 4 * np.pi)        
    #print samples
    
    rng = np.linspace(0.07, 0.13)
    plt.hist(samples[1500:], bins = 100, normed=True)
    plt.plot(rng, np.exp(stats.gamma.logpdf(rng, 0.1 + 500, 0, 1.0 / (0.1 + 500*9.42))))

    plt.show()
    # Make more room for the actual plot
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    #rng = np.linspace(0.07,0.13)
    #plt.plot(rng, stats.gamma.pdf(rng, 0.1 + 500, 0.1 + 500*0.0992))
    #ax1 = plt.subplot(gs[0])
    #plt.hold(True)
    #plt.plot(test_range, real_func(test_range) )
    #plt.fill_between(test_range, real_func(test_range) - noise, real_func(test_range) + noise, color=4*[0.5])
    
    #plt.plot([0, 4*np.pi], 2*[y_star], color='black')
    #plt.fill_between([0, 4 * np.pi], 2*[y_star - epsilon], 2*[y_star + epsilon], color=4*[0.5])
    #plt.scatter(samples, sample_values, marker='.')
    
    #ax2 = plt.subplot(gs[1], sharex=ax1)
    #plt.hist(samples, bins=100, range=(0, 4 * np.pi), normed=True)
        
    #plt.show()
#    
if __name__ == '__main__':
    ASL_ABC()
    #cProfile.run('ASL_ABC()')