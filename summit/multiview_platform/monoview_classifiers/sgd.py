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
#
#
#
from sklearn.linear_model import SGDClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomUniform

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SGD"


class SGD(SGDClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `SGDClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_


    """

    def __init__(self, random_state=None, loss='hinge',
                 penalty='l2', alpha=0.0001, max_iter=5, tol=None, **kwargs):
        SGDClassifier.__init__(self,
                               loss=loss,
                               penalty=penalty,
                               alpha=alpha,
                               max_iter=max_iter,
                               tol=tol,
                               random_state=random_state
                               )
        self.param_names = ["loss", "penalty", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [['log', 'modified_huber'],
                         ["l1", "l2", "elasticnet"],
                         CustomUniform(loc=0, state=1), [random_state]]
        self.weird_strings = {}
