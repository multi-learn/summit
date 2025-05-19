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
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
# Version:
# -------
#
# * multiview_generator version = 0.0.1
#
# Licence:
# -------
#
# License: New BSD License
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
