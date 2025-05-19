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
from sklearn.metrics import fbeta_score as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, beta=2.0, average="micro", **kwargs):
    score = metric(y_true, y_pred, beta=beta, average=average, **kwargs)
    return score


def get_scorer(beta=2.0, average="micro", **kwargs):
    return make_scorer(metric, greater_is_better=True, beta=beta,
                       average=average, **kwargs)


def get_config(beta=2.0, average="micro", **kwargs):
    config_string = "F-beta score using beta: {}, average: {}, {} (higher is better)".format(
        beta, average, kwargs)
    return config_string
