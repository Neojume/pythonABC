import numpy as np
from random import uniform, gauss
import matplotlib.pyplot as plt
from numpy import linalg
from collections import defaultdict
import kernels
    
def adaptive_KDE(x_star, X, t, kernel=kernels.gaussian, h=1, alpha=0):
    '''
    Returns the kernel density estimate at x_star using the given kernel and 
    bandwidth on the given data.
    
    Parameters
    ----------
    x_star : estimation location
    X : coordinates of samples
    t : sample values
    kernel : kernel to use for the estimate, default epanechnikov
    bandwidth : bandwidth of the kernel, default 1
    '''
    # Calculate adaptive bandwidths
    bandwidth = defaultdict(lambda : alpha)
    for i, x1 in enumerate(X):
        for x2 in X:
            u = linalg.norm(x1 - x2) / h
            bandwidth[i] +=  (kernel(u) / h) * (x1 - x2) ** 2    
    
    weights = []
    for x in X:
        u = linalg.norm(x_star - x) / bandwidth[i]
        weight = kernel(u) / bandwidth[i]
        weights.append(weight[0,0])
            
    weights = np.matrix(weights).T    
    weighted = np.multiply(weights, t)
    
    N = np.log(np.sum(weights))
    #print N
    
    mean = np.sum(weighted) / np.exp(N)
    #std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) / N)
    
    # Use unbiased estimator
    V2 = np.log(np.sum(np.square(weights)))
    #std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) * N / (N**2.0 - V2))
    sigma = np.log(np.sum(np.multiply(weights, np.square(mean - t))))
    std = 0.5 * (sigma - N - V2)
    return mean, np.exp(std), N
    
def adaptive_KDE_old(x_star, X, t, kernel=kernels.gaussian, h=1.0, alpha = 0):

    # Calculate adaptive bandwidths
    bandwidth = defaultdict(lambda : alpha)
    for i, x1 in enumerate(X):
        for x2 in X:
            u = linalg.norm(x1 - x2) / h
            bandwidth[i] +=  (kernel(u) / h) * (x1 - x2) ** 2
    
    weights = []
    for i, x in enumerate(X):
        u = linalg.norm(x_star - x) / bandwidth[i]
        weight = kernel(u) / bandwidth[i]
            
        weights.append(weight[0,0])
            
    weights = np.matrix(weights).T    
    weighted = np.multiply(weights, t)
    
    N = np.sum(weights)
    mean = np.sum(weighted) / N       
    #std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) / N)
    
    # Unbiased estimator of the variance
    V2 = np.sum(np.square(weights))
    std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) * N / ((N ** 2.0) - V2))
    
    return mean, std, N
    
def kernel_regression(x_star, X, t, kernel=kernels.gaussian, bandwidth=1):
    '''
    Returns the kernel regression estimate at x_star using the given kernel and 
    bandwidth on the given data.
    
    Parameters
    ----------
        x_star : Estimate location
        
        X : Coordinates of samples
        
        t : Sample values
        
        kernel : kernel to use for the estimate, default Gaussian
        
        bandwidth : bandwidth of the kernel, default 1
    '''
    t = np.array(t)
    weights = []
    for x in np.array(X):
        u = linalg.norm(x_star - x) / bandwidth
        weight = kernel(u) / bandwidth
        weights.append(weight)
            
    weights = np.matrix(weights).T    
    weighted = np.multiply(weights, t)
    
    N = np.log(np.sum(weights))

    
    mean = np.sum(weighted) / np.exp(N)       
    #std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) / N)
    
    # Use unbiased estimator
    V2 = np.log(np.sum(np.square(weights)))
    #std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) * N / (N**2.0 - V2))
    summ = np.sum(np.multiply(weights, np.square(mean - t)))
    sigma = np.log(summ)
    std = 0.5 * (sigma - N - V2)
    if np.isnan(std):
        print 'mean', mean
        print 'std', std
        print 'N', N
        print 'V2', V2
        print 'sigma', sigma
        print 'summ', summ
        print weights.T
        print ''
    return mean, np.exp(std), N
   
def MSE(predictions, targets):
    '''
    Returns the mean squared error.
    '''
    return np.mean((predictions - targets) ** 2)    

def biggest_gap(x_coords, limits):
    gap = 0.0
    prev = limits[0]
    for x in sorted(x_coords + [limits[1]]):        
        diff = x - prev
        if diff > gap:
            gap = diff
        prev = x
    return gap

def real_func(x):
    return 20 * np.sin(x / 1.2) + 2 * x

if __name__ == '__main__':
    
    num_samples = 30
    noise_level = 2
    limits = [0.0, 4 * np.pi]
    
    noise = [gauss(0.0, noise_level) for i in range(num_samples)]
    #x_coords = [uniform(limits[0] , limits[1]) for i in range(num_samples)]
    
    # So that there is a gap in between
    d = limits[1] - limits[0]
    x_coords = [uniform(limits[0], limits[0] + d / 4.0) for i in range(num_samples/2)]
    x_coords += [uniform(limits[1] - d / 4.0, limits[1]) for i in range(num_samples/2)]
    
    gap = biggest_gap(x_coords, limits)
    
    X = np.matrix(x_coords).T
    t = real_func(X) + np.matrix(noise).T

    precision = 50

    test_range = np.linspace(limits[0], limits[1], precision)
    bandwidth_range = np.linspace(gap, limits[1] - limits[0], precision) 
    
    errors = []   
    
    bandwidth = 0.6
    
    # Reset the means and stds
    means = []
    stds = []
    Ns = []
    
    # Compute the kernel density estimates
    for i in test_range:
        mu, sigma, N = kernel_regression(i, X, t, bandwidth=bandwidth)
        means.append(mu)
        stds.append(sigma)
        Ns.append(N)

    # Convert to np arrays
    means = np.array(means)
    stds = np.array(stds)
    
    # Compute the error and add it to the list
    error = MSE(means, real_func(test_range))
    errors.append(error)
    
    
    # Plot the fit
    ax1 = plt.subplot(221)
    ax1.set_ylabel('KDE')
    plt.plot(test_range, real_func(test_range), color='r')
    plt.plot(test_range, means, color='y')
    
    plt.ylim(-40, 60)
    plt.xlim(limits[0], limits[1])
    plt.fill_between(test_range, (means - 2*stds).flat, (means + 2*stds).flat, color=[0.7,0.7,0.7,0.7])
    plt.scatter(X.flat, t.flat)
    
    plt.subplot(222, sharex=ax1)
    plt.plot(test_range, Ns, label='Ns')
    plt.plot(test_range, stds, label='stds')
    plt.legend()
    
    # Reset the means and stds
    means = []
    stds = []
    Ns = []
    
    # Compute the kernel density estimates
    for i in test_range:
        mu, sigma, N = adaptive_KDE(i, X, t, h=bandwidth)
        means.append(mu)
        stds.append(sigma)
        Ns.append(N)

    # Convert to np arrays
    means = np.array(means)
    stds = np.array(stds)

    ax2 = plt.subplot(223, sharex=ax1)
    ax2.set_ylabel('Adaptive KDE')
    plt.plot(test_range, real_func(test_range), color='r')
    plt.plot(test_range, means, color='y')
    
    plt.xlim(limits[0], limits[1])
    plt.ylim(-40, 60)
    plt.fill_between(test_range, (means - 2*stds).flat, (means + 2*stds).flat, color=[0.7,0.7,0.7,0.7])
    #plt.fill_between(test_range, real_func(test_range) - 2*noise_level, real_func(test_range) + 2*noise_level, color=[0.7,0.7,0.7,0.7])
    #plt.fill_between(test_range, (means - stds).flat, (means + stds).flat, color='gray')
    plt.scatter(X.flat, t.flat)
    #plt.plot(bandwidth_range, errors)
    
    plt.subplot(224, sharex=ax2)
    plt.plot(test_range, Ns, label='Ns')
    plt.plot(test_range, stds, label='stds')
    plt.legend()
    
    plt.show()
    