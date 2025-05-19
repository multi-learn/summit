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
from .additions.early_fusion_from_monoview import BaseEarlyFusion
from ..utils.hyper_parameter_search import CustomUniform

classifier_class_name = "EarlyFusionSGD"


class EarlyFusionSGD(BaseEarlyFusion):

    def __init__(self, random_state=None, loss='hinge',
                 penalty='l2', alpha=0.0001, max_iter=5, tol=None, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="sgd", loss=loss,
                 penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol, **kwargs)
        self.param_names = ["loss", "penalty", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [['log', 'modified_huber'],
                         ["l1", "l2", "elasticnet"],
                         CustomUniform(loc=0, state=1), [random_state]]
        self.weird_strings = {}