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
import os
import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from .. import metrics
from ..monoview.monoview_utils import BaseMonoviewClassifier, get_accuracy_graph
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint
from ..utils.base import base_boosting_estimators

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "Adaboost"


class Adaboost(AdaBoostClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `AdaBoostClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier>`_

    """

    def __init__(self, random_state=None, n_estimators=50, 
                 estimator=None, estimator_config=None,   **kwargs):
        estimator = BaseMonoviewClassifier.get_base_estimator(self,
                                                                   estimator, estimator_config)
        AdaBoostClassifier.__init__(self,
                                    random_state=random_state,
                                    n_estimators=n_estimators,
                                    estimator=estimator,
                                    algorithm="SAMME"
                                    )
        self.param_names = ["n_estimators", "estimator"]
        self.classed_params = ["estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                         base_boosting_estimators]
        self.weird_strings = {"estimator": "class_name"}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None
        self.estimator_config = estimator_config

    def fit(self, X, y, sample_weight=None):
        begin = time.time()
        AdaBoostClassifier.fit(self, X, y, sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = X.shape
        self.base_predictions = np.array(
            [estim.predict(X) for estim in self.estimators_])
        self.metrics = np.array([self.plotted_metric.score(pred, y) for pred in
                                 self.staged_predict(X)])
        return self

    def predict(self, X):
        begin = time.time()
        pred = AdaBoostClassifier.predict(self, X)
        end = time.time()
        self.pred_time = end - begin
        self.step_predictions = np.array(
            [step_pred for step_pred in self.staged_predict(X)])
        return pred

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multi_class=False):  # pragma: no cover
        interpretString = ""
        interpretString += self.get_feature_importance(directory,
                                                       base_file_name,
                                                       feature_ids)
        interpretString += "\n\n Estimator error | Estimator weight\n"
        interpretString += "\n".join(
            [str(error) + " | " + str(weight / sum(self.estimator_weights_)) for
             error, weight in
             zip(self.estimator_errors_, self.estimator_weights_)])
        step_test_metrics = np.array(
            [self.plotted_metric.score(y_test, step_pred) for step_pred in
             self.step_predictions])
        get_accuracy_graph(step_test_metrics, "Adaboost",
                           os.path.join(directory,
                                        base_file_name + "test_metrics.png"),
                           self.plotted_metric_name, set="test")
        np.savetxt(os.path.join(directory, base_file_name + "test_metrics.csv"),
                   step_test_metrics,
                   delimiter=',')
        np.savetxt(
            os.path.join(directory, base_file_name + "train_metrics.csv"),
            self.metrics, delimiter=',')
        np.savetxt(os.path.join(directory, base_file_name + "times.csv"),
                   np.array([self.train_time, self.pred_time]), delimiter=',')
        return interpretString
