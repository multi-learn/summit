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
from sklearn.ensemble import RandomForestClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "RandomForest"


class RandomForest(RandomForestClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_


    """

    def __init__(self, random_state=None, n_estimators=10,
                 max_depth=None, criterion='gini', **kwargs):

        RandomForestClassifier.__init__(self,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        criterion=criterion,
                                        random_state=random_state
                                        )
        self.param_names = ["n_estimators", "max_depth", "criterion",
                            "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomRandint(low=1, high=10),
                         ["gini", "entropy"], [random_state]]
        self.weird_strings = {}

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multiclass=False):

        interpret_string = ""
        interpret_string += self.get_feature_importance(directory,
                                                        base_file_name,
                                                        feature_ids)
        return interpret_string
