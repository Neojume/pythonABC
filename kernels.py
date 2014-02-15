# -*- coding: utf-8 -*-
"""
Implementation of some kernel functions

@author: steven
"""
import numpy as np
    
def epanechnikov(u):
    if np.abs(u) <= 1.0:
        return 0.75 * (1 - np.square(u))
    return 0.0

def triangular(u):
    if np.abs(u) <= 1.0:
        return (1 - np.abs(u))
    return 0.0

def logistic(u):
    return 1.0 / (np.exp(u) + 2.0 + np.exp(-u))
    
def log_logistic(u):
    v = [2.0, u, -u]
    m = max(v)
    return - np.log(np.exp(v-m).sum())

def gaussian(u):
    return np.exp(-0.5 * np.square(u)) / np.sqrt(2 * np.pi)
    
def log_gaussian(u):
    return -0.5 * np.square(u) - np.log(np.sqrt(2 * np.pi))
