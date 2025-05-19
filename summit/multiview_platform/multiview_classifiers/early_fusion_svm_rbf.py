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

classifier_class_name = "EarlyFusionSVMRBF"


class EarlyFusionSVMRBF(BaseEarlyFusion):

    def __init__(self, random_state=None, C=1.0, **kwargs):
        BaseEarlyFusion.__init__(self, random_state=random_state,
                                 monoview_classifier="svm_rbf", C=C, **kwargs)
        self.param_names = ["C", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1), [random_state]]