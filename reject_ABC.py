import numpy as np
import distributions as distr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def reject_ABC(problem, epsilon, num_samples, verbose=True):
    y_star = problem.y_star

    simulator = problem.simulator

    prior = problem.prior
    prior_args = problem.prior_args

    samples = []
    
    sim_calls = 0
    
    for i in range(num_samples):
        if verbose:
            if i % 200 == 0: 
                print i

        error = epsilon + 1.0
        while  error > epsilon:
            # Sample x from the (uniform) prior
            x = prior.rvs(*prior_args)
            
            # Perform simulation
            y = simulator(x)
            sim_calls += 1

            # Calculate error
            error = abs(y_star - y)
            
        # Accept the sample
        samples.append(x)

    return samples, sim_calls

if __name__ == '__main__':
    from problems import toy_problem

    problem = toy_problem()

    samples, sim_calls = reject_ABC(problem, 0.05, 10000)

    print 'sim_calls', sim_calls

    precision = 100
    test_range = np.linspace(0.07, 0.13, 100)
    plt.plot(test_range, problem.real_posterior(test_range))
    plt.hist(samples[1500:], 100, normed=True, alpha=0.5)
    plt.show()    
