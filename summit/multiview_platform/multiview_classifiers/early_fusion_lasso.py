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
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint

classifier_class_name = "EarlyFusionLasso"


class EarlyFusionLasso(BaseEarlyFusion):

    def __init__(self, random_state=None, alpha=1.0,
                 max_iter=10, warm_start=False, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=None, alpha=alpha,
                                 max_iter=max_iter,
                                 warm_start=warm_start, **kwargs)
        self.param_names = ["max_iter", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=30--0),
                         CustomUniform(), [random_state]]