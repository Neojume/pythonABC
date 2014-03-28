# -*- coding: utf-8 -*-
"""
Class containing different probability distributions with methods to calculate
their pdf, logpdf and a method to sample from the distributions.

@author: Steven
"""

import collections
import numpy as np
import pylab as pp
import scipy.special as special
from scipy.integrate import quad


class proportional(object):

    '''
    Distribution class that computes the pdf and cdf values using a function
    that is proportional to the true probability density.

    Works only for 1D densities.
    '''

    def __init__(self, proportional_function):
        '''
        Initialize this distribution class by precalculating the normalization
        constant.

        Arguments
        ---------
        proportional : function
            Function that is proportional to the computed distribution
        '''

        self.proportional_function = proportional_function
        self.normalization, _ = quad(
            self.proportional_function,
            -np.inf,
            np.inf)

    def pdf(self, x):
        return self.proportional_function(x) / self.normalization

    def cdf(self, x):
        if isinstance(x, collections.Iterable):
            val = np.array([quad(self.pdf, -np.inf, i)[0] for i in x])
        else:
            val = quad(self.pdf, -np.inf, x)[0]
        return val


class gamma(object):

    '''
    The Gamma distribution.
    '''

    @staticmethod
    def pdf(x, alpha, beta):
        return np.exp(gamma.logpdf(x, alpha, beta))

    @staticmethod
    def logpdf(x, alpha, beta):
        if beta > 0:
            return alpha * np.log(beta) - \
                special.gammaln(alpha) + (alpha - 1) * np.log(x) - beta * x
        else:
            assert False, "Beta is zero"

    @staticmethod
    def cdf(x, alpha, beta):
        return special.gammainc(alpha, x * beta)

    @staticmethod
    def rvs(alpha, beta, N=1):
        return np.random.gamma(alpha, 1.0 / beta, N)


class exponential(object):

    '''
    The exponential distribution.
    '''

    @staticmethod
    def pdf(x, beta):
        return np.exp(exponential.logpdf(x, beta))

    @staticmethod
    def logpdf(x, beta):
        if beta > 0:
            return np.log(beta) - beta * x
        else:
            assert False, "Beta is zero"

    @staticmethod
    def rvs(beta, N=1):
        return np.random.exponential(1.0 / beta, N)


class lognormal(object):

    '''
    The log-normal distribution. When X is normally distributed,
    then Y = exp(X) is log-normally distributed.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(lognormal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        if type(x) == np.ndarray:
            if sigma > 0:
                small = np.log(0.5 + 0.5 * special.erf((np.log(1e-6) - mu) /
                               (np.sqrt(2.0) * sigma)))
                I = pp.find(x > 1e-6)
                log_x = np.log(x[I])
                lp = small * np.ones(x.shape)
                lp[I] = -log_x - 0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
                    0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
            else:
                I = pp.find(x == mu)
                lp = -np.inf * np.ones(x.shape)
                lp[I] = 0
        else:
            if sigma > 0:
                if x > 1e-6:
                    log_x = np.log(x)
                    lp = - log_x - 0.5 * np.log(2.0 * np.pi) - \
                        np.log(sigma) - \
                        0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
                else:
                    lp = np.log(0.5 + 0.5 * special.erf((np.log(1e-6) - mu) /
                                (np.sqrt(2.0) * sigma)))
            else:
                if x == mu:
                    lp = 0
                else:
                    lp = -np.inf
        return lp

    @staticmethod
    def rvs(mu, sigma):
        return np.random.lognormal(mu, sigma)


class logitnormal(object):

    '''
    The logit-normal distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(logitnormal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        return - np.log(x) - np.log(1 - x) - \
            0.5 * np.log(2.0 * np.pi * sigma ** 2) - \
            0.5 * ((np.log(x) - np.log(1 - x) - mu) / sigma) ** 2

    @staticmethod
    def rvs(mu, sigma):
        # TODO: Implement
        pass


class uniform(object):

    '''
    The 1D uniform distribution.
    '''

    @staticmethod
    def pdf(x, a=0, b=1):
        #if a <= x <= b:
        return (np.all([a <= x, x <= b], axis=0)) / (b - a)
        #return 0.0

    @staticmethod
    def logpdf(x, a=0, b=1):
        if a <= x <= b:
            return -np.log(b - a)
        return - np.inf

    @staticmethod
    def rvs(a=0, b=1, N=1):
        return np.random.uniform(a, b, N)


class uniform_nd(object):

    '''
    The n-dimensional uniform distribution.
    '''

    @staticmethod
    def pdf(x, p1, p2):
        float(np.all(x > p1) and np.all(x < p2)) / np.prod(p2 - p1)

    @staticmethod
    def logpdf(x, p1, p2):
        if np.all(x > p1) and np.all(x < p2):
            return -np.sum(np.log(p2 - p1))
        return -np.inf

    @staticmethod
    def rvs(p1, p2):
        return np.random.uniform(p1, p2)


class normal(object):

    '''
    The 1D normal (or Gaussian) distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(normal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
            0.5 * ((x - mu) ** 2) / (sigma ** 2)

    @staticmethod
    def rvs(mu, sigma, N=1):
        return np.random.normal(mu, sigma, N)


class multivariate_normal(object):

    '''
    The multivariate normal (or Gaussian) distribution.
    '''

    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(multivariate_normal.logpdf(x, mu, sigma))

    @staticmethod
    def logpdf(x, mu, sigma):
        m = np.matrix(x - mu)
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        try:
            k = x.shape[-1]
        except AttributeError:
            # Floats have no shape attribute
            k = 1
        return -0.5 * (m * inv * m.T + k * np.log(2 * np.pi) + np.log(det))

    @staticmethod
    def rvs(mu, sigma, N=1):
        return np.random.multivariate_normal(mu, sigma, N).ravel()

# TODO: Implement more multidimensional distributions
