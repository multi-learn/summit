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
from summit.multiview_platform.monoview_classifiers.additions.SVCClassifier import \
    SVCClassifier
from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import \
    CustomUniform, CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SVMPoly"


class SVMPoly(SVCClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_

    Here, it is the polynomial kernel version
    """

    def __init__(self, random_state=None, C=1.0, degree=3, **kwargs):
        SVCClassifier.__init__(self,
                               C=C,
                               kernel='poly',
                               degree=degree,
                               random_state=random_state
                               )
        self.param_names = ["C", "degree", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1),
                         CustomRandint(low=2, high=30), [random_state]]
