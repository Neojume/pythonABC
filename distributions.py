# -*- coding: utf-8 -*-
"""
Class containing different probability distributions with methods to calculate 
their pdf, logpdf and a method to sample from the distributions.

Created on Mon Feb 03 17:59:08 2014

@author: Steven
"""

import numpy as np
import scipy.special as special
import pylab as pp


class gamma(object):
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
    def rvs(alpha, beta, N=1):
        return np.random.gamma(alpha, 1.0 / beta, N)


class normal(object):
    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(normal.logpdf(x, mu, sigma))
        
    @staticmethod        
    def logpdf(x, mu, sigma):
        return - 0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
                 0.5 * ((x - mu) ** 2) / (sigma ** 2)
        
    @staticmethod
    def rvs(mu, sigma, N=1):
        return np.random.normal(mu, sigma, N)

        
class exponential(object):
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

        
class uniform(object):
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
        
        
class lognormal(object):
    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(lognormal.logpdf(x, mu, sigma))
        
    @staticmethod        
    def logpdf(x, mu, sigma):
        if x.__class__ == np.ndarray:
            if sigma > 0:
                small = np.log(0.5 + 0.5 * special.erf((np.log(1e-6) - mu) / \
                        (np.sqrt(2.0) * sigma)))
                I = pp.find(x > 1e-6)
                log_x = np.log(x[I])
                lp = small*np.ones(x.shape)
                lp[I] = -log_x - 0.5 * np.log(2.0 * np.pi) - np.log(sigma) - \
                        0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
            else:
                I = pp.find(x == mu)
                lp = -np.inf*np.ones(x.shape)
                lp[I] = 0
        else:
            if sigma > 0:
                if x > 1e-6:
                    log_x = np.log(x)
                    lp    = - log_x - 0.5 * np.log(2.0 * np.pi) - \
                            np.log(sigma) - \
                            0.5 * ((log_x - mu) ** 2) / (sigma ** 2)
                else:
                    lp = small
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
    @staticmethod
    def pdf(x, mu, sigma):
        return np.exp(logitnormal.logpdf(x, mu, sigma))
        
    @staticmethod        
    def logpdf(x, mu, sigma):
        return - np.log(x) - np.log(1 - x) - 0.5 * np.log(2.0 * np.pi * sigma ** 2) - 0.5 * ((np.log(x) - np.log(1 - x) - mu) / sigma) ** 2
        
    @staticmethod
    def rvs(mu, sigma):
        pass
            
