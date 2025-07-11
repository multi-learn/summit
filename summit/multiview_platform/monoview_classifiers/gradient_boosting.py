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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import metrics
from ..monoview.monoview_utils import BaseMonoviewClassifier, get_accuracy_graph
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "GradientBoosting"


class CustomDecisionTreeGB(DecisionTreeClassifier):
    def predict(self, X, check_input=True):
        y_pred = DecisionTreeClassifier.predict(self, X,
                                                check_input=check_input)
        return y_pred.reshape((y_pred.shape[0], 1)).astype(float)


class GradientBoosting(GradientBoostingClassifier, BaseMonoviewClassifier):
    """
     This class is an adaptation of scikit-learn's `GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_


     """

    def __init__(self, random_state=None, loss="exponential", max_depth=1.0,
                 n_estimators=100,
                 init=CustomDecisionTreeGB(max_depth=1),
                 **kwargs):
        GradientBoostingClassifier.__init__(self,
                                            loss=loss,
                                            max_depth=max_depth,
                                            n_estimators=n_estimators,
                                            init=init,
                                            random_state=random_state
                                            )
        self.param_names = ["n_estimators", "max_depth"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=50, high=500),
                         CustomRandint(low=1, high=10), ]
        self.weird_strings = {}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None

    def fit(self, X, y, sample_weight=None, monitor=None):
        begin = time.time()
        GradientBoostingClassifier.fit(self, X, y, sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = X.shape
        self.base_predictions = np.array(
            [estim[0].predict(X) for estim in self.estimators_])
        self.metrics = np.array(
            [self.plotted_metric.score(pred, y) for pred in
             self.staged_predict(X)])
        # self.bounds = np.array([np.prod(
        #     np.sqrt(1 - 4 * np.square(0.5 - self.estimator_errors_[:i + 1]))) for i
        #                         in range(self.estimator_errors_.shape[0])])
        return self

    def predict(self, X):
        begin = time.time()
        pred = GradientBoostingClassifier.predict(self, X)
        end = time.time()
        self.pred_time = end - begin
        if X.shape != self.train_shape:
            self.step_predictions = np.array(
                [step_pred for step_pred in self.staged_predict(X)])
        return pred

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multi_class=False):
        interpretString = ""
        if multi_class:
            return interpretString
        else:
            interpretString += self.get_feature_importance(directory,
                                                           base_file_name,
                                                           feature_ids)
            step_test_metrics = np.array(
                [self.plotted_metric.score(y_test, step_pred) for step_pred in
                 self.step_predictions])
            get_accuracy_graph(step_test_metrics, "AdaboostClassic",
                               directory + "test_metrics.png",
                               self.plotted_metric_name, set="test")
            get_accuracy_graph(self.metrics, "AdaboostClassic",
                               directory + "metrics.png",
                               self.plotted_metric_name)
            np.savetxt(
                os.path.join(directory, base_file_name + "test_metrics.csv"),
                step_test_metrics,
                delimiter=',')
            np.savetxt(
                os.path.join(directory, base_file_name + "train_metrics.csv"),
                self.metrics,
                delimiter=',')
            np.savetxt(os.path.join(directory, base_file_name + "times.csv"),
                       np.array([self.train_time, self.pred_time]),
                       delimiter=',')
            return interpretString
