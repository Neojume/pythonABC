# -*- coding: utf-8 -*-
"""
Implementation of some kernel functions

@author: steven
"""
import numpy as np

__all__ = ['epanechnikov',
           'tricube',
           'triweight',
           'triangular',
           'logistic',
           'log_logistic',
           'gaussian',
           'log_gaussian',
           'param_gaussian',
           'circle_spike',
           'exponential']

def epanechnikov(u):
    return 0.75 * (1 - np.square(np.clip(u, -1, 1)))


def tricube(u):
    return np.array(abs(u) <= 1.0, int) * 70.0 / 81.0 * \
        (1 - np.abs(u) ** 3) ** 3


def triweight(u):
    return (35.0 / 32.0) * pow(1 - np.square(np.clip(u, -1, 1)), 3.0)


def triangular(u):
    return (1 - np.abs(np.clip(u, -1, 1)))


def logistic(u):
    return 1.0 / (np.exp(u) + 2.0 + np.exp(-u))


def log_logistic(u):
    v = [2.0, u, -u]
    m = max(v)
    return - np.log(np.exp(v - m).sum())


def gaussian(u):
    # sigma = 1
    return np.exp(-0.5 * np.square(u)) / np.sqrt(2 * np.pi)


# NOTE: this only works for 1 / x sigmas where x is an integer.
def param_gaussian(sigma):
    def _param_gaussian(u):
        return np.exp(- np.square(u) / (2 * sigma ** 2.0)) / (np.sqrt(2 * np.pi) * sigma)
    kernel = _param_gaussian
    kernel.func_name = 'param_gaussian_sigma' + str(int(1.0/sigma))
    return param_gaussian

def log_gaussian(u):
    return -0.5 * np.square(u) - np.log(np.sqrt(2 * np.pi))


def circle_spike(u):
    return 3.0 * (1.0 - np.sqrt(np.abs(np.clip(u, -1, 1)))) ** 2


def exponential(u):
    return 0.5 * (np.e - np.exp(np.abs(np.clip(u, -1, 1))))
