import distributions as distr


def reject_ABC(problem, num_samples, epsilon, verbose=True):
    '''
    Performs rejection ABC on the given problem.

    Parameters
    ----------
        problem: The problem to solve An instance of the ABC_Problem class.

        num_samples: The number of samples to sample.

        epsilon: The error margin or epsilon-tube.

        verbose: If set to true iteration number as well as number of
            simulation calls will be printed.

    Returns
    -------
    A tuple: samples, sim_calls

        samples: list of samples

        sim_calls: list of simulation calls needed for each sample
    '''
    y_star = problem.y_star

    simulator = problem.simulator

    prior = problem.prior
    prior_args = problem.prior_args

    samples = []

    sim_calls = []

    for i in range(num_samples):
        current_sim_calls = 0
        error = epsilon + 1.0

        while error > epsilon:
            # Sample x from the (uniform) prior
            x = prior.rvs(*prior_args)

            # Perform simulation
            y = simulator(x)
            current_sim_calls += 1

            # Calculate error
            error = abs(y_star - y)

        # Accept the sample
        samples.append(x)
        sim_calls.append(current_sim_calls)

        if verbose:
            if i % 200 == 0:
                print i, current_sim_calls, sum(sim_calls)

    return samples, sim_calls

if __name__ == '__main__':
    from problems import toy_problem
    import matplotlib.pyplot as plt
    import numpy as np

    problem = toy_problem()

    samples, sim_calls = reject_ABC(problem, 10000, 0.05)

    print 'sim_calls', sum(sim_calls)

    precision = 100
    test_range = np.linspace(0.07, 0.13, 100)
    post = problem.true_posterior
    post_args = problem.true_posterior_args
    plt.plot(test_range, np.exp(post.logpdf(test_range, *post_args)))
    plt.hist(samples[1500:], 100, normed=True, alpha=0.5)
    plt.show()
