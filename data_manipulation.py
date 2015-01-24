import os.path
import os
import pickle

__all__ = ['ABCData',
           'get_filename',
           'save',
           'load']

class ABCData(object):

    '''
    Storage class for abc-data.
    '''

    def __init__(self, algorithm):
        '''
        Create a data-container for this combination.
        '''

        self.algorithm = type(algorithm).__name__
        self.alg_args = algorithm.get_parameters()
        self.problem = type(algorithm.problem).__name__

        self.num_data = 0
        self.list_of_samples = []
        self.list_of_accepts = []
        self.list_of_sim_calls = []
        self.list_of_sim_locs = []

    def add_datum(self, algorithm):
        '''
        Add data to the database.
        '''
        self.list_of_samples.append(algorithm.samples)
        self.list_of_sim_calls.append(algorithm.sim_calls)
        try:
            self.list_of_accepts.append(algorithm.accepted)
        except AttributeError:
            # Algorithm has no accepted attribute
            pass
        try:
            self.list_of_sim_locs.append(algorithm.xs)
        except AttributeError:
            pass

        self.num_data += 1


def get_filename(algorithm):
    '''
    Creates a unique filename for this algorithm.

    Arguments
    ---------
    algorithm : instance of an `ABC_Algorithm`
        The algorithm instance to generate a name for.
    '''

    return str(algorithm) + '.abc'


def load(algorithm, data_dir=False):
    '''
    Loads the results for the given algorithm with given parameters for
    the given problem. If there are no results for these combinations None is
    returned.

    Parameters
    ----------
    algorithm :
        An instance of an ABC algorithm
    data_dir : boolean
        Whether there is a directory `data` where the file is stored

    Returns
    -------
    data : ABCData or None
        The loaded data. Or None if there is no data.
    '''

    filename = get_filename(algorithm)

    if data_dir:
        path = os.path.join(os.getcwd(), 'data', filename)
    else:
        path = os.path.join(os.getcwd(), filename)

    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    else:
        return None


def save(algorithm, data_dir=False):
    '''
    Saves the results for the given algorithm with given parameters for
    the given problem.

    Parameters
    ----------
    algorithm :
        An instance of an ABC algorithm
    data_dir : boolean
        Whether there is a directory `data` where the file is stored
    '''

    filename = get_filename(algorithm)

    if data_dir:
        path = os.path.join(os.getcwd(), 'data', filename)
    else:
        path = os.path.join(os.getcwd(), filename)

    if os.path.isfile(path):
        # If file exists open and append to the existing database
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        # Otherwise create a new database
        data = ABCData(algorithm)

    data.add_datum(algorithm)

    with open(path, 'wb') as f:
        pickle.dump(data, f)
