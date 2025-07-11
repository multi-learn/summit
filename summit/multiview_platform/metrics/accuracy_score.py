# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
"""Functions :
 score: to get the accuracy score
 get_scorer: returns a sklearn scorer for grid search
"""

from sklearn.metrics import accuracy_score as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    """Arguments:
    y_true: real labels
    y_pred: predicted labels

    Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    Weighted accuracy score for y_true, y_pred"""
    score = metric(y_true, y_pred, **kwargs)
    return score


def get_scorer(**kwargs):
    """Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    A weighted sklearn scorer for accuracy"""
    return make_scorer(metric, greater_is_better=True,
                       **kwargs)


def get_config(**kwargs):
    config_string = "Accuracy score using {}, (higher is better)".format(
        kwargs)
    return config_string
