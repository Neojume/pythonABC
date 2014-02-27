import numpy as np
import distributions as distr
import matplotlib.pyplot as plt

def conditional_error(alphas, u, tau, M):
    if u <= tau:
        return float(np.sum(alphas < u)) / M
    else:
        return float(np.sum(alphas >= u)) / M

def ASL_ABC(problem, num_samples, epsilon, ksi, S0, delta_S, verbose=False):
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

    Returns
    -------
    samples, sim_calls, rate : tuple
        samples: list of samples
        sim_calls: list of simulation calls needed for each sample
        rate: the acceptance rate
    '''

    y_star = problem.y_star

    # TODO: make adaptive
    eye = np.identity(1)

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
    
    samples = []    

    sim_calls = []
    accepted = 0
            
    for i in range(num_samples):
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
            error = np.mean([e * conditional_error(alphas, e, tau, M) \
                    for e in np.linspace(0,1,E)])

            if error < ksi:
                break
        
        current_sim_calls = 2 * S

        if distr.uniform.rvs(0.0, 1.0) <= tau:
            accepted += 1
            theta = theta_p
            log_theta = np.log(theta_p)
        
        # Add the sample to the set of samples
        samples.append(theta)
        sim_calls.append(current_sim_calls)

        if verbose:
            if i % 200 == 0:
                print i, current_sim_calls, sum(sim_calls)

    return samples, sim_calls, float(accepted) / num_samples

if __name__ == '__main__':
    from problems import toy_problem

    problem = toy_problem()
    samples, sim_calls, rate = ASL_ABC(problem, 10000, 0, 0.05, 5, 10)

    print 'sim_calls', sum(sim_calls)
    print 'acceptance rate', rate

    rng = np.linspace(0.07, 0.13)
    plt.hist(samples[1500:], bins=100, normed=True)
    plt.plot(rng, problem.real_posterior(rng))

    plt.show()
