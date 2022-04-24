#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
import from 50b745c1d18d5c4b01d9d00e406b5fdaab3515ea @ KamLearn

Utility routines
"""

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import sys
import logging
import numpy as np

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['add_constant_feature',
           'fill_missing_with_mean',
           'decode_nfv']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def add_constant_feature(D):
    """ add ones at the first column of the matrix"""

    return np.c_[np.ones(D.shape[0]), D]

def fill_missing_with_mean(D, default=0.0):
    """ fill missing value with the means of non-missing values in the column"""

    for i in range(D.shape[1]):
        if np.any(np.isnan(D[:, i])):
            v = np.mean(D[np.isfinite(D[:, i]), i])
            if np.isnan(v):
                v = default
            D[np.isnan(D[:, i]), i] = v

    return D

def decode_nfv(nfvstr, nf):
    """ parse the string for a list of feature domain sizes"""

    try:
        nfv = np.fromstring(nfvstr, dtype=int, sep=':')

        if len(nfv) == 1:
            nfv = nfv * nf
        elif len(nfv) != nf:
            raise ValueError
        if np.any(nfv < 0) or np.any(nfv == 1):
            raise ValueError
    except ValueError:
        sys.exit("Illegal specfication of the numbers of feature values")

    return nfv

#==============================================================================
# Module initialization
#==============================================================================

# init logging system ---------------------------------------------------------

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

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
