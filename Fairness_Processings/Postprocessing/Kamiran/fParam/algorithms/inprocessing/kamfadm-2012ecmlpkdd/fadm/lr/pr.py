#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two Class logistic regression module with Prejudice Remover
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np
from scipy.optimize import fmin_cg
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['LRwPRType4']

#==============================================================================
# Constants
#==============================================================================

EPSILON = 1.0e-10
SIGMOID_RANGE = np.log((1.0 - EPSILON) / EPSILON)
N_S = 1
N_CLASSES = 2

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def sigmoid(x, w):
    s = np.clip(np.dot(w, x), -SIGMOID_RANGE, SIGMOID_RANGE)

    return 1.0 / (1.0 + np.exp(-s))


#==============================================================================
# Classes
#==============================================================================

class LRwPR(BaseEstimator, ClassifierMixin):
    """ Two class LogisticRegression with Prejudice Remover"""

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        if C < 0.0:
            raise TypeError
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.eta = eta
        self.minor_type = 0
        self.f_loss_ = np.inf

    def predict(self, X):
        """ predict classes"""

        return np.argmax(self.predict_proba(X), 1)

class LRwPRPredictProbaType2Mixin(LRwPR):
    """ mixin for singe type 2 likelihood"""

    def predict_proba(self, X):
        """ predict probabilities"""

        # add a constanet term
        s = np.atleast_1d(np.squeeze(np.array(X)[:, -self.n_s_]).astype(int))
        if self.fit_intercept:
            X = np.c_[np.atleast_2d(X)[:, :-self.n_s_], np.ones(X.shape[0])]
        else:
            X = np.atleast_2d(X)[:, :-self.n_s_]
        coef = self.coef_.reshape(self.n_sfv_, self.n_features_)

        proba = np.empty((X.shape[0], N_CLASSES))
        proba[:, 1] = [sigmoid(X[i, :], coef[s[i], :])
                       for i in range(X.shape[0])]
        proba[:, 0] = 1.0 - proba[:, 1]

        return proba

class LRwPRFittingType1Mixin(LRwPR):
    """ Fitting Method Mixin"""

    def init_coef(self, itype, X, y, s):
        """ set initial weight"""

        if itype == 0:
            # clear by zeros
            self.coef_ = np.zeros(self.n_sfv_ * self.n_features_,
                                  dtype=np.float)
        elif itype == 1:
            # at random
            self.coef_ = np.random.randn(self.n_sfv_ * self.n_features_)

        elif itype == 2:
            # learned by standard LR
            self.coef_ = np.empty(self.n_sfv_ * self.n_features_,
                                  dtype=np.float)
            coef = self.coef_.reshape(self.n_sfv_, self.n_features_)

            clr = LogisticRegression(C=self.C, penalty='l2',
                                     fit_intercept=False)
            clr.fit(X, y)

            coef[:, :] = clr.coef_
        elif itype == 3:
            # learned by standard LR
            self.coef_ = np.empty(self.n_sfv_ * self.n_features_,
                                  dtype=np.float)
            coef = self.coef_.reshape(self.n_sfv_, self.n_features_)

            for i in range(self.n_sfv_):
                clr = LogisticRegression(C=self.C, penalty='l2',
                                         fit_intercept=False)
                clr.fit(X[s == i, :], y[s == i])
                coef[i, :] = clr.coef_
        else:
            raise TypeError

    def fit(self, X, y, ns=N_S, itype=0, **kwargs):
        """ train this model"""

        # rearrange input arguments
        s = np.atleast_1d(np.squeeze(np.array(X)[:, -ns]).astype(int))
        if self.fit_intercept:
            X = np.c_[np.atleast_2d(X)[:, :-ns], np.ones(X.shape[0])]
        else:
            X = np.atleast_2d(X)[:, :-ns]

        # check optimization parameters
        if not 'disp' in kwargs:
            kwargs['disp'] = False
        if not 'maxiter' in kwargs:
            kwargs['maxiter'] = 100

        # set instance variables
        self.n_s_ = ns
        self.n_sfv_ = np.max(s) + 1
        self.c_s_ = np.array([np.sum(s == si).astype(np.float)
                              for si in range(self.n_sfv_)])
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        # optimization
        self.init_coef(itype, X, y, s)
        self.coef_ = fmin_cg(self.loss,
                             self.coef_,
                             fprime=self.grad_loss,
                             args=(X, y, s),
                             **kwargs)

        # get final loss
        self.f_loss_ = self.loss(self.coef_, X, y, s)

class LRwPRObjetiveType4Mixin(LRwPR):
    """ objective function of logistic regression with prejudice remover"""

    def loss(self, coef_, X, y, s):

        coef = coef_.reshape(self.n_sfv_, self.n_features_)

#        print >> sys.stderr, "loss:", coef[0, :], coef[1, :]

        ### constants

        # sigma = Pr[y=0|x,s] = sigmoid(w(s)^T x)
        p = np.array([sigmoid(X[i, :], coef[s[i], :])
                      for i in range(self.n_samples_)])

        # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
        q = np.array([np.sum(p[s == si])
                      for si in range(self.n_sfv_)]) / self.c_s_

        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si)
        r = np.sum(p) / self.n_samples_

        ### loss function

        # likelihood
        # \sum_{x,s,y in D} y log(sigma) + (1 - y) log(1 - sigma)
        l = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        # fairness-aware regularizer
        # \sum_{x,s in D} \
        #    sigma(x,x)       [log(rho(s))     - log(pi)    ] + \
        #    (1 - sigma(x,s)) [log(1 - rho(s)) - log(1 - pi)]
        f = np.sum(p * (np.log(q[s]) - np.log(r))
             + (1.0 - p) * (np.log(1.0 - q[s]) - np.log(1.0 - r)))

        # l2 regularizer
        reg = np.sum(coef * coef)

        l = -l + self.eta * f + 0.5 * self.C * reg
#        print >> sys.stderr, l
        return l

    def grad_loss(self, coef_, X, y, s):
        """ first derivative of loss function"""

        coef = coef_.reshape(self.n_sfv_, self.n_features_)
        l_ = np.empty(self.n_sfv_ * self.n_features_)
        l = l_.reshape(self.n_sfv_, self.n_features_)
#        print >> sys.stderr, "grad_loss:", coef[0, :], coef[1, :]

        ### constants
        # prefix "d_": derivertive by w(s)

        # sigma = Pr[y=0|x,s] = sigmoid(w(s)^T x)
        # d_sigma(x,s) = d sigma / d w(s) = sigma (1 - sigma) x
        p = np.array([sigmoid(X[i, :], coef[s[i], :])
                      for i in range(self.n_samples_)])
        dp = (p * (1.0 - p))[:, np.newaxis] * X

        # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
        # d_rho(s) = \sum_{(xi,si)in D st si=s} d_sigma(xi,si) / #D[s]
        q = np.array([np.sum(p[s == si])
                      for si in range(self.n_sfv_)]) / self.c_s_
        dq = np.array([np.sum(dp[s == si, :], axis=0)
                       for si in range(self.n_sfv_)]) \
                       / self.c_s_[:, np.newaxis]

        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si) / #D
        # d_pi = \sum_{(xi,si)in D} d_sigma(xi,si) / #D
        r = np.sum(p) / self.n_samples_
        dr = np.sum(dp, axis=0) / self.n_samples_

        # likelihood
        # l(si) = \sum_{x,y in D st s=si} (y - sigma(x, si)) x
        for si in range(self.n_sfv_):
            l[si, :] = np.sum((y - p)[s == si][:, np.newaxis] * X[s == si, :],
                              axis=0)


        f1 = (np.log(q[s]) - np.log(r)) \
             - (np.log(1.0 - q[s]) - np.log(1.0 - r))
        f2 = (p - q[s]) / (q[s] * (1.0 - q[s]))
        f3 = (p - r) / (r * (1.0 - r))
        f4 = f1[:, np.newaxis] * dp \
            + f2[:, np.newaxis] * dq[s, :] \
            - np.outer(f3, dr)
        f = np.array([np.sum(f4[s == si, :], axis=0)
                      for si in range(self.n_sfv_)])

        # l2 regularizer
        reg = coef

        # sum
        l[:, :] = -l + self.eta * f + self.C * reg

        return l_

class LRwPRType4\
    (LRwPRObjetiveType4Mixin,
     LRwPRFittingType1Mixin,
     LRwPRPredictProbaType2Mixin):
    """ Two class LogisticRegression with Prejudice Remover"""

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        super(LRwPRType4, self).\
            __init__(C=C, eta=eta,
                     fit_intercept=fit_intercept, penalty=penalty)

        self.coef_ = None
        self.mx_ = None
        self.n_s_ = 0
        self.n_sfv_ = 0
        self.minor_type = 4

#==============================================================================
# Module initialization
#==============================================================================

# init logging system

logger = logging.getLogger('fadm')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

def _test():
    """ test function for this module"""

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script

if __name__ == '__main__':
    _test()
