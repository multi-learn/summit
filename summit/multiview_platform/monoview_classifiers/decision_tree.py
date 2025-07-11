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
from sklearn.tree import DecisionTreeClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "DecisionTree"


class DecisionTree(DecisionTreeClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `DecisionTreeClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_


    """

    def __init__(self, random_state=None, max_depth=None,
                 criterion='gini', splitter='best', **kwargs):

        DecisionTreeClassifier.__init__(self,
                                        max_depth=max_depth,
                                        criterion=criterion,
                                        splitter=splitter,
                                        random_state=random_state
                                        )
        self.param_names = ["max_depth", "criterion", "splitter",
                            'random_state']
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         ["gini", "entropy"],
                         ["best", "random"], [random_state]]
        self.weird_strings = {}

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multiclass=False):
        interpretString = "First feature : \n\t{} <= {}\n".format(
            feature_ids[self.tree_.feature[0]],
            self.tree_.threshold[0])
        interpretString += self.get_feature_importance(directory,
                                                       base_file_name,
                                                       feature_ids)
        return interpretString
