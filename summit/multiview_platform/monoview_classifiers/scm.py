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
from pyscm.scm import SetCoveringMachineClassifier as scm

import numpy as np

from ..monoview.monoview_utils import BaseMonoviewClassifier
from ..utils.hyper_parameter_search import CustomRandint, CustomUniform


# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


# class Decis
classifier_class_name = "SCM"

class SCM(scm, BaseMonoviewClassifier):
    """
    SCM  Classifier
    Parameters
    ----------
    random_state (default : None)
    model_type : string (default: "conjunction")
    max_rules : int number maximum of rules (default : 10)
    p : float value(default : 0.1 )

    kwarg : others arguments

    Attributes
    ----------
    param_names

    distribs

    classed_params

    weird_strings

    """

    def __init__(self, random_state=None, model_type="conjunction",
                 max_rules=10, p=0.1, **kwargs):
        """

        Parameters
        ----------
        random_state
        model_type
        max_rules
        p
        kwargs
        """
        super(SCM, self).__init__(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rules,
            p=p
        )
        self.param_names = ["model_type", "max_rules", "p", "random_state"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [random_state]]
        self.classed_params = []
        self.weird_strings = {}

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        self.n_features = X.shape[1]
        scm.fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params)
        self.feature_importances_ = np.zeros(self.n_features)
        # sum the rules importances :
        # rules_importances = estim.get_rules_importances() #activate it when pyscm will implement importance
        rules_importances = np.ones(len(
            self.model_.rules))  # delete it when pyscm will implement importance
        for rule, importance in zip(self.model_.rules, rules_importances):
            self.feature_importances_[rule.feature_idx] += importance
        self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     return False in any case
    #     """
    #     return False

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multi_class=False):
        interpret_string = self.get_feature_importance(directory,
                                                       base_file_name,
                                                       feature_ids)
        interpret_string += "Model used : " + str(self.model_)
        return interpret_string



def paramsToSet(nIter, random_state):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append(
            {"model_type": random_state.choice(["conjunction", "disjunction"]),
             "max_rules": random_state.randint(1, 15),
             "p": random_state.random_sample()})
    return paramsSet
