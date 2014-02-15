import distributions as distr
import numpy as np

class ABC_Problem(object):
    y_star = None

    prior = None
    prior_args = None

    proposal = None
    proposal_args = None

    theta_init = None

    def simulator(theta):
        raise NotImplemented
    
    def real_posterior(x):
        raise NotImplemented

class toy_problem(ABC_Problem):
    '''
    The exponential toy problem of Meeds and Welling.
    '''

    def __init__(self, y_star=9.42, N=500):
        self.y_star = y_star
        
        self.N = N

        self.prior = distr.gamma
        self.prior_args = [0.1, 0.1]
        
        self.proposal = distr.lognormal
        self.proposal_args = [0.1]

        self.theta_init = 1.0
    
    def simulator(self, theta):
        return np.mean(distr.exponential.rvs(theta, self.N))

    def real_posterior(self, x):
        return np.exp(distr.gamma.logpdf(x, 
            self.prior_args[0] + self.N, 
            self.prior_args[1] + self.N * self.y_star))
