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
from ..utils.base import base_boosting_estimators

# from ..utils.dataset import get_v

classifier_class_name = "EarlyFusionAdaboost"


class EarlyFusionAdaboost(BaseEarlyFusion):

    def __init__(self, random_state=None, n_estimators=50,
                 estimator=None, base_estimator_config=None, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="adaboost",
                                 n_estimators= n_estimators,
                                 estimator=estimator,
                                 base_estimator_config=base_estimator_config, **kwargs)
        self.param_names = ["n_estimators", "estimator"]
        self.classed_params = ["estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                         base_boosting_estimators]
        self.weird_strings = {"estimator": "class_name"}
