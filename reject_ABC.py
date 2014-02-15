import numpy as np
import distributions as distr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def real_func(x):
    return np.sin(x / 1.2) + 0.10 * x
    #return norm.pdf(x, np.pi, np.pi)

def simulator(x, noise=0.2):
    return real_func(x) + distr.normal.rvs(0, noise)

def real_posterior(x, noise, y_star, epsilon):
    mu = real_func(x)
    return distr.normal.pdf(y_star, mu, noise) * distr.uniform.pdf(mu, 0, 4 * np.pi)# - norm.cdf(y_star - epsilon, mu, noise)

if __name__ == '__main__':
    
    num_samples = 20000
    noise = 0.3
    
    # The value we would like to replicate
    y_star = 1.3
    
    # Epsilon tube
    epsilon = 0.1
    
    samples = []
    sample_values = []
    
    sim_calls = 0
    
    for i in range(num_samples):
        if i % 200 == 0: print i
        error = epsilon + 1.0
        while  error > epsilon:
            # Sample x from the (uniform) prior
            x = distr.uniform.rvs(0.0, 4 * np.pi)
            
            # Perform simulation
            y = simulator(x, noise)
            sim_calls += 1

            # Calculate error
            error = abs(y_star - y)
            
        # Accept the sample
        samples.append(x)
        sample_values.append(y)

    print 'sim_calls', sim_calls

    test_range = np.linspace(0, 4 * np.pi, 100)        

    # Make more room for the actual plot
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = plt.subplot(gs[0])
    plt.plot(test_range, real_func(test_range) )
    plt.fill_between(test_range, real_func(test_range) - noise, real_func(test_range) + noise, color=4*[0.5])
    
    plt.plot([0, 4*np.pi], 2*[y_star], color='black')
    plt.fill_between([0, 4 * np.pi], 2*[y_star - epsilon], 2*[y_star + epsilon], color=4*[0.5])
    #plt.scatter(samples, sample_values, marker='.')
    
    ax2 = plt.subplot(gs[1], sharex=ax1)
    #plt.hist(samples, bins=100, range=(0, 4 * np.pi), normed=True)
    #m, bins = np.histogram(samples, bins=200, range=(0, 4 * np.pi), normed=True)
    #m = m / float(np.sum(m))
    #print m
    
    post = real_posterior(test_range, noise, y_star, epsilon)
    post = post / (np.sum(post)   * (4 * np.pi / 100))
    plt.plot(test_range, post)  
    m, bins = np.histogram(samples, bins=200, range=(0, 4 * np.pi), normed=True)
    plt.plot(bins[1:], m, color='red')
    plt.show()
    
