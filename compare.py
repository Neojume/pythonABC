'''
Implements currently one distance measure.
'''

def variation_distance3(samples, problem, num_bins=200):
    diff = np.zeros(num_samples)

    cdf = problem.true_posterior.cdf
    args = problem.true_posterior_args

    for i in xrange(num_samples):
        bins, edges = np.histogram(samples[:i+1], num_bins)
        ecdf = np.cumsum(bins) / float(i + 1)

        width = edges[1] - edges[0]
        diff[i] = np.sum(np.abs(ecdf - cdf(np.array(edges[1:]), *args))) * width

    return diff

def variation_distance(samples, problem, num_bins=100):
    diff = np.zeros(num_samples)
    missed_mass = np.zeros(num_samples)

    cdf = problem.true_posterior.cdf
    args = problem.true_posterior_args

    for i in xrange(num_samples):
        bins, edges = np.histogram(samples[:i+1], num_bins, normed=True)

        w = float(edges[1] - edges[0])
        edges = np.array(edges)
        bins2 = (cdf(edges[1:], *args) - cdf(edges[:-1], *args)) / w

        vals = np.isfinite(bins2)

        missed_mass[i] = 1.0 - w * sum(bins2[vals])
        diff[i] = 0.5 * (sum(abs(bins[vals] - bins2[vals])) * w + missed_mass[i])


    return diff

def variation_distance2(samples, problem, num_bins=100):
    diff = np.zeros(num_samples)

    cdf = problem.true_posterior.cdf
    args = problem.true_posterior_args

    for i in xrange(num_samples):
        if i % 100 == 0:
            print i
        #bins, edges = np.histogram(samples[:i+1], num_bins, normed=True)

        sorted_samples = np.sort(samples[:i+1]) #list(sorted(samples[:i+1]))

        dist = 0.0
        prev_ecdf = 0.0
        prev_x = 0.0
        prev_cdf = 0.0

        for j, x in enumerate(sorted_samples[1:]):
            if x == prev_x:
                continue

            cur_ecdf = float(j + 1) / num_samples
            ecdf_diff = cur_ecdf - prev_ecdf
            cur_cdf = cdf(x, *args)
            cdf_diff = cur_cdf - prev_cdf

            dist += abs(cdf_diff - ecdf_diff) * (x - prev_x)
            
            prev_ecdf = cur_ecdf
            prev_cdf = cur_cdf
            prev_x = x

        diff[i] = dist

    return diff


if __name__ == '__main__2':
    import numpy as np
    import matplotlib.pyplot as plt
    
    from SL_ABC import SL_ABC
    from problems import toy_problem

    import time

    problem = toy_problem()

    num_samples = 4000
    num_bins = 100

    cdf = problem.true_posterior.cdf
    args = problem.true_posterior_args


    samples, sim_calls, _ = SL_ABC(problem, num_samples, 0, 5, True)
    
    samples = samples[1500:]
    num_samples -= 1500

    #Calculate ecdf
    ecdf = [float(n + 1) / num_samples for n in xrange(num_samples)]
    ecdf_x = list(sorted(samples))

    dist = 0
    prev_ecdf = 0.0
    prev_x = 0.0
    prev_cdf = 0.0
    for i, x in enumerate(ecdf_x[1:]):
        if x == prev_x:
            continue

        cur_ecdf = float(i + 1) / num_samples
        ecdf_diff = cur_ecdf - prev_ecdf
        cur_cdf = cdf(x, *args)
        cdf_diff = cur_cdf - prev_cdf

        dist += abs(cdf_diff - ecdf_diff) * (x - prev_x)
        
        prev_ecdf = cur_ecdf
        prev_cdf = cur_cdf
        prev_x = x

    print dist

    plt.step(ecdf_x, ecdf)
    rng = np.linspace(ecdf_x[0], ecdf_x[-1])
    plt.plot(rng, cdf(rng, *args))
    plt.show()
    t = time.time()
    dist = variation_distance(samples, problem)
    print time.time() - t
    t = time.time()
    dist2 = variation_distance3(samples, problem)
    print time.time() - t

    bins, edges = np.histogram(samples, num_bins, normed=True)

    width = float(edges[1] - edges[0])
    edges = np.array(edges)
    bins2 = (cdf(edges[1:], *args) - cdf(edges[:-1], *args)) / width

    plt.bar(edges[:-1], bins, width=width, alpha=0.5, color='r')
    plt.bar(edges[:-1], bins2, width=width, alpha=0.5, color='g')
    plt.bar(edges[:-1], abs(bins2 - bins), width=width, alpha=0.5, color='b')
    plt.plot(edges, np.exp(problem.true_posterior.logpdf(edges, *args)))

    plt.show()
    plt.plot(dist, label='1')
    plt.plot(dist2, label='2')
    plt.legend()
    plt.show()
    raw_input()

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    from SL_ABC import SL_ABC
    from ASL_ABC import ASL_ABC
    from marginal_ABC import marginal_ABC
    from reject_ABC import reject_ABC
    from problems import toy_problem

    problem = toy_problem()

    num_samples = 3000

    ax1 = plt.subplot(121)
    ax1.set_xlabel('Number of samples')
    ax1.set_ylabel('Error')
    ax2 = plt.subplot(122, sharey=ax1)
    ax2.set_xlabel('Number of simulation calls')

    samples, sim_calls, _ = ASL_ABC(problem, num_samples, 0, 0.05, 10, 5, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='ASL')
    ax2.plot(np.cumsum(sim_calls), dist, label='ASL')

    samples, sim_calls, _ = SL_ABC(problem, num_samples, 0, 10, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='SL')
    ax2.plot(np.cumsum(sim_calls), dist, label='SL')

    samples, sim_calls, _ = marginal_ABC(problem, num_samples, 0.05, 10, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='marginal')
    ax2.plot(np.cumsum(sim_calls), dist, label='marginal')
    
    samples, sim_calls = reject_ABC(problem, num_samples, 0.1, True)
    dist = variation_distance(samples, problem)
    ax1.plot(dist, label='rejection')
    ax2.plot(np.cumsum(sim_calls), dist, label='rejection')

    ax1.legend()
    ax2.legend()

    plt.show()
