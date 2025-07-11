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

from sklearn.svm import SVC

from .additions.jumbo_fusion_utils import BaseJumboFusion
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint

classifier_class_name = "SVMJumboFusion"


class SVMJumboFusion(BaseJumboFusion):

    """
    This classifier learns monoview classifiers on each view and then uses an
    SVM on their decisions to aggregate them.
    """

    def __init__(self, random_state=None, classifiers_names=None,
                 classifier_configs=None, nb_cores=1, weights=None,
                 nb_monoview_per_view=1, C=1.0, kernel="rbf", degree=2,
                 rs=None):
        self.need_probas = False
        BaseJumboFusion.__init__(self, random_state,
                                 classifiers_names=classifiers_names,
                                 classifier_configs=classifier_configs,
                                 nb_cores=nb_cores, weights=weights,
                                 nb_monoview_per_view=nb_monoview_per_view,
                                 rs=rs)
        self.param_names += ["C", "kernel", "degree"]
        self.distribs += [CustomUniform(), ["rbf", "poly", "linear"],
                          CustomRandint(2, 5)]
        self.aggregation_estimator = SVC(C=C, kernel=kernel, degree=degree)
        self.C = C
        self.kernel = kernel
        self.degree = degree

    def set_params(self, C=1.0, kernel="rbf", degree=1, **params):
        super(SVMJumboFusion, self).set_params(**params)
        self.C = C
        self.degree = degree
        self.kernel = kernel
        self.aggregation_estimator.set_params(
            C=C, kernel=kernel, degree=degree)
        return self
