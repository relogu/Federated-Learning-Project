#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:11:01 2021

@author: relogu
"""

import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             homogeneity_score, normalized_mutual_info_score,
                             rand_score)

# definition of the metrics used
nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score
ran = rand_score
homo = homogeneity_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
