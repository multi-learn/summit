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
from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomRandint
from ..monoview_classifiers.gradient_boosting import CustomDecisionTreeGB

classifier_class_name = "EarlyFusionGB"


class EarlyFusionGB(BaseEarlyFusion):

    def __init__(self, random_state=None, loss="exponential", max_depth=1.0,
                 n_estimators=100,
                 init=CustomDecisionTreeGB(max_depth=1),
                 **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="gradient_boosting",
                                 loss=loss, max_depth=max_depth,
                                 n_estimators=n_estimators, init=init, **kwargs)
        self.param_names = ["n_estimators", "max_depth"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=50, high=500),
                         CustomRandint(low=1, high=10), ]