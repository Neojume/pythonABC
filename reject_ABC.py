import data_manipulation as dm


def reject_ABC(problem, num_samples, epsilon, verbose=True, save=True):
    '''
    Performs rejection ABC on the given problem.

    Parameters
    ----------
        problem : ABC_Problem instance
            The problem to solve An instance of the ABC_Problem class.
        num_samples : int
            The number of samples to sample.
        epsilon : float
            The error margin or epsilon-tube.
        verbose : bool
            If set to true iteration number as well as number of
            simulation calls will be printed.
        save : bool
            If True will save the result to a (possibly exisisting) database

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
            # TODO: use statistics and arbitrary compare method
            error = abs(y_star - y)

        # Accept the sample
        samples.append(x)
        sim_calls.append(current_sim_calls)

        if verbose:
            if i % 200 == 0:
                print i, current_sim_calls, sum(sim_calls)

    if save:
        dm.save(reject_ABC, [epsilon], problem, (samples, sim_calls))

    return samples, sim_calls
