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
from summit.multiview_platform.monoview_classifiers.additions.SVCClassifier import \
    SVCClassifier
from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomUniform

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SVMLinear"


class SVMLinear(SVCClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_

    Here, it is the linear kernel version
    """

    def __init__(self, random_state=None, C=1.0, **kwargs):
        SVCClassifier.__init__(self,
                               C=C,
                               kernel='linear',
                               random_state=random_state
                               )
        self.param_names = ["C", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1), [random_state]]
