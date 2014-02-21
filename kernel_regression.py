import numpy as np
from random import uniform, gauss
import matplotlib.pyplot as plt
from numpy import linalg
from collections import defaultdict
import kernels
import scipy.optimize as opt

def adaptive_kernel_regression(x_star, X, t, h, kernel=kernels.gaussian, alpha=0):
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
            
    weights = np.array(weights)    
    weighted = weights * t
    
    N = np.log(np.sum(weights))
    
    mean = np.sum(weighted) / np.exp(N)
    
    # Use unbiased estimator
    V2 = np.log(np.sum(np.square(weights)))
    sigma = np.log(np.sum(weights * np.square(mean - t)))
    std = 0.5 * (sigma - N - V2)
    return mean, np.exp(std), N
    
def MLCV(h, X, kernel):
    ans = 0
    n = len(X)

    for i, x1 in enumerate(X):
        for j, x2 in enumerate(X):
            if i == j:
                continue

            u = linalg.norm(x2 - x1) / h
            ans += np.log(kernel(u)) 
        ans -= - np.log((n - 1) * h)

    # NOTE: Return minus the answer, because scipi only minimizes
    return ans

def kernel_regression(x_star, X, t, h, kernel=kernels.gaussian):
    '''
    Returns the kernel regression estimate at x_star using the given kernel and 
    bandwidth on the given data.
    
    Parameters
    ----------
    x_star : Estimate location
    
    X : Coordinates of samples
    
    t : Sample values
    
    kernel : kernel to use for the estimate, default Gaussian
    
    h : bandwidth of the kernel, default MLVC estimate
    '''

    t = np.array(t)
    weights = []
    for x in np.array(X):
        u = linalg.norm(x_star - x) / h
        #u = (x_star - x) / h
        weight = kernel(u) / h
        weights.append(weight)
            
    weights = np.array(weights)    
    weighted = weights * t
    
    N = np.log(np.sum(weights))

    
    mean = np.sum(weighted) / np.exp(N)       
    
    # Use unbiased estimator
    V2 = np.log(np.sum(np.square(weights)))
    summ = np.sum(weights * np.square(mean - t))
    sigma = np.log(summ)
    #std = 0.5 * (sigma + N - V2)
    std = 0.5 * (sigma - N)
    if np.isnan(std):
        print 't', t
        print 'X', X
        print 'x_star', x_star
        print 'mean', mean
        print 'std', std
        print 'N', N
        print 'V2', V2
        print 'sigma', sigma
        print 'summ', summ
        print weights.T
        print weighted
        print ''
    return mean, np.exp(std), N

def kernel_regression_old(x_star, X, t, h, kernel=kernels.gaussian):

    t = np.array(t)
    weights = []
    for x in np.array(X):
        u = linalg.norm(x_star - x) / h
        weight = kernel(u) / h
        weights.append(weight)
            
    weights = np.matrix(weights).T    
    weighted = np.multiply(weights, t)
    
    N = np.sum(weights)
    mean = np.sum(weighted) / N       

    #std = np.sum(np.multiply(weights, np.square(mean - t))) / N
    std = np.sqrt(np.sum(np.multiply(weights, np.square(mean - t))) / N)
    return mean, std, N
    

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
    
    num_samples = 300
    noise_level = 2
    limits = [0.0, 4 * np.pi]
    
    noise = [gauss(0.0, noise_level) for i in range(num_samples)]
    x_coords = [uniform(limits[0] , limits[1]) for i in range(num_samples)]
    
    # So that there is a gap in between
    d = limits[1] - limits[0]
    #x_coords = [uniform(limits[0], limits[0] + d / 4.0) for i in range(num_samples/2)]
    #x_coords += [uniform(limits[1] - d / 4.0, limits[1]) for i in range(num_samples/2)]
    
    gap = biggest_gap(x_coords, limits)
    
    X = np.array(x_coords)
    t = real_func(X) + np.array(noise)

    precision = 50

    test_range = np.linspace(limits[0], limits[1], precision)
    bandwidth_range = np.linspace(gap, limits[1] - limits[0], precision) 
    
    errors = []   
    
    bandwidth = 0.5
    
    # Reset the means and stds
    means = []
    stds = []
    Ns = []
    
    # Compute the kernel density estimates
    for i in test_range:
        mu, sigma, N = kernel_regression(i, X, t, bandwidth)
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
    ax1 = plt.subplot(211)
    ax1.set_ylabel('KRE')
    plt.plot(test_range, real_func(test_range), color='r')
    plt.plot(test_range, means, color='y')
    
    plt.ylim(-40, 60)
    plt.xlim(limits[0], limits[1])
    plt.fill_between(test_range, (means - 2*stds).flat, (means + 2*stds).flat, color=[0.7,0.7,0.7,0.7])
    plt.scatter(X.flat, t.flat)
    
    plt.subplot(212, sharex=ax1)
    plt.plot(test_range, Ns, label='Ns')
    plt.plot(test_range, stds, label='stds')
    plt.legend()
    plt.show()
    
