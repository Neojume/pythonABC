import numpy as np
import distributions as distr
import matplotlib.pyplot as plt

def SL_ABC(problem, S, epsilon, num_samples):

    # Make local copies of problem parameters for speed

    y_star = problem.y_star
    
    # Identity matrix of same dimension as y_star
    # TODO: Make adaptive
    eye = np.identity(1)    
    
    prior = problem.prior 
    prior_args = problem.prior_args
    
    proposal = problem.proposal
    proposal_args = problem.proposal_args
    
    simulator = problem.simulator

    samples = []
    sim_calls = 0
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
        sim_calls += 2 * S

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
    
    return samples, float(accepted) / num_samples, sim_calls

if __name__ == '__main__':
    from problems import toy_problem

    problem = toy_problem()
    
    samples, acceptance_rate, sim_calls = SL_ABC(problem, 50, 0, 10000)

    print 'sim_calls', sim_calls
    print 'acceptance rate', acceptance_rate

    def real_posterior(x, N=500):
        return np.exp(distr.gamma.logpdf(x, 0.1 + N, 0.1 + N * 9.42))

    precision = 100
    test_range = np.linspace(0.07, 0.13, 100)
    plt.plot(test_range, real_posterior(test_range))
    plt.hist(samples[1500:], 100, normed=True, alpha=0.5)
    plt.show()    

