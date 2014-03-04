import os.path
import os
import pickle

class ABCData(object):
    def __init__(self):
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
    # Create filename
    filename = type(problem).__name__ + '_' + algorithm.__name__
    for arg in alg_args:
        filename += '_' + str(arg)
    filename += '.abc'
    return filename

def save(algorithm, alg_args, problem, datum):
    '''
    Save the results from an ABC run.
    '''

    filename = get_filename(algorithm, alg_args, problem)

    # Check if file already exists, if so add to the database
    path = os.path.join(os.getcwd(), 'data', filename)
    if os.path.isfile(path):
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
    else:
        data = ABCData()

    data.add_datum(datum)

    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()

