"""
    This experiment compares performance between: 
    (1) assuming a distribution 
    (2) SAA (non-parametric) 
    (3) DRMPC
"""

import numpy as np
import cvxpy as cp


# == Sampled-Averaged Apporixmaitons  ==
"""
SAA is the non-parametric (ignores assumed distribution)
"""