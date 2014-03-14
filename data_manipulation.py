import os.path
import os
import pickle


class ABCData(object):

    def __init__(self, algorithm, alg_args, problem):
        '''
        Create a data-container for this combination.
        '''

        self.algorithm = algorithm
        self.alg_args = alg_args
        self.problem = problem

        self.num_data = 0
        self.list_of_samples = []
        self.list_of_accepts = []
        self.list_of_sim_calls = []

    def add_datum(self, data):
        '''
        Add data to the database.
        '''
        self.list_of_samples.append(data[0])
        self.list_of_sim_calls.append(data[1])
        if len(data) > 2:
            self.list_of_accepts.append(data[2])
        self.num_data += 1


def get_filename(algorithm, alg_args, problem):
    '''
    Creates a unique filename for this algorithm + problem combination.
    '''
    # Create filename
    filename = type(problem).__name__ + '_' + algorithm.__name__
    for arg in alg_args:
        filename += '_' + str(arg)
    filename += '.abc'
    return filename


def load(algorithm, alg_args, problem):
    '''
    Loads the results for the given algorithm with given parameters for
    the given problem. If there are no results for these combinations None is
    returned.

    Parameters
    ----------
    algorithm :
        An instance of an ABC algorithm

    alg_args : list
        List of arguments for the given algorithm.

    problem : ABC_Problem
        The problem to solve

    Returns
    -------
    data : ABCData or None
        The loaded data. Or None if there is no data.
    '''

    filename = get_filename(algorithm, alg_args, problem)

    path = os.path.join(os.getcwd(), 'data', filename)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None


def save(algorithm, alg_args, problem, datum):
    '''
    Saves the results for the given algorithm with given parameters for
    the given problem.

    Parameters
    ----------
    algorithm :
        An instance of an ABC algorithm

    alg_args : list
        List of arguments for the given algorithm.

    problem : ABC_Problem
        The problem to solve

    datum : tuple
        The data to save
    '''

    filename = get_filename(algorithm, alg_args, problem)

    # Check if file already exists, if so add to the database
    path = os.path.join(os.getcwd(), 'data', filename)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = ABCData(algorithm, alg_args, problem)

    data.add_datum(datum)

    with open(path, 'wb') as f:
        pickle.dump(data, f)
