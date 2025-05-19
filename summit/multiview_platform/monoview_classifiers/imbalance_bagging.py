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
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from ..utils.base import base_boosting_estimators
from ..utils.hyper_parameter_search import CustomRandint, CustomUniform

classifier_class_name = "ImbalanceBagging"

class ImbalanceBagging(BaseMonoviewClassifier, BalancedBaggingClassifier):

    def __init__(self, random_state=None, estimator="DecisionTreeClassifier",
                 n_estimators=10, sampling_strategy="auto",
                 replacement=False, base_estimator_config=None):
        estimator = self.get_base_estimator(estimator,
                                                 base_estimator_config)
        super(ImbalanceBagging, self).__init__(random_state=random_state, estimator=estimator,
                                         n_estimators=n_estimators,
                                         sampling_strategy=sampling_strategy,
                                         replacement=replacement)

        self.param_names = ["n_estimators", "estimator", "sampling_strategy",]
        self.classed_params = ["estimator"]
        self.distribs = [CustomRandint(low=1, high=50),
                         base_boosting_estimators,
                         ["auto"]]
        self.weird_strings = {"base_estimator": "class_name"}
        self.base_estimator_config = base_estimator_config



