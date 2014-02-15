import kernels
import numpy as np
import distributions as distr
from numpy import linalg
from random import seed

import matplotlib.pyplot as plt

def simulator(theta, N=500):
    return np.mean(distr.exponential.rvs(theta, N))

def real_posterior(x, N=500):
    return np.exp(distr.gamma.logpdf(x, 0.1 + N, 0.1 + N * 9.42))

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

if __name__ == '__main__':
    # Parameters
    num_samples = 10000
    y_star = 9.42
    epsilon = 0.05
    S = 20
    
    prior = distr.gamma
    prior_args = [0.1, 0.1]
    
    proposal = distr.lognormal
    proposal_args = [0.1]
    
    log_theta = 0.0
    theta = 1.0
    
    samples = []
    sim_calls = 0
    accepted = 0

    for i in xrange(num_samples):
        # Propose a new theta
        theta_p = proposal.rvs(log_theta, *proposal_args)
        log_theta_p = np.log(theta_p)
        
        x = []
        x_p = []
        
        diff = []
        diff_p = []
        
        # Get S samples and approximate marginal likelihood
        for s in xrange(S):
            new_x = simulator(theta)
            new_x_p = simulator(theta_p)
            sim_calls += 2
            
            # Compute the P(y | x, theta) for these samples
            u = linalg.norm(new_x - y_star) / epsilon
            u_p = linalg.norm(new_x_p - y_star) / epsilon
            
            diff.append(kernels.log_gaussian(u / epsilon))
            diff_p.append(kernels.log_gaussian(u_p / epsilon))
            
            x.append(new_x)
            x_p.append(new_x_p)
        
        numer = prior.logpdf(theta_p, *prior_args) + proposal.logpdf(theta, log_theta_p, *proposal_args)
        denom = prior.logpdf(theta, *prior_args) + proposal.logpdf(theta_p, log_theta, *proposal_args)
        
        diff_term = logsumexp(np.array(diff_p)) - logsumexp(np.array(diff))
        
        log_alpha = min(0.0, (numer - denom) + diff_term) 
         
        if distr.uniform.rvs(0, 1) <= np.exp(log_alpha):
            accepted += 1
            theta = theta_p
            log_theta = log_theta_p
            
        samples.append(theta)
    
    print 'sim_calls', sim_calls
    print 'acceptance ratio', accepted / float(num_samples)

    # Create plots of how close we are
    rng = np.linspace(0.07, 0.13, 100)
    plt.hist(samples[1500:], bins = 100, normed=True)
    plt.plot(rng, real_posterior(rng))
    plt.show()
