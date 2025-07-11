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
import unittest
import numpy as np
import pandas as pd
import os

from summit.multiview_platform.monoview.monoview_utils import MonoviewResult
from summit.multiview_platform.multiview.multiview_utils import MultiviewResult

from summit.multiview_platform.result_analysis.metric_analysis import get_metrics_scores, init_plot, get_fig_size, sort_by_test_score


class Test_get_metrics_scores(unittest.TestCase):

    def test_simple(self):
        metrics = {"accuracy_score*": {}, "f1_score": {}}
        results = [MonoviewResult(0,
                                  "ada",
                                  "0",
                                  {"accuracy_score*": [0.9, 0.95],
                                   "f1_score":[0.91, 0.96]}, "", "", "", "", "", 0, 0, {})]
        metrics_scores, class_met = get_metrics_scores(metrics,
                                                       results, [])
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score*"], pd.DataFrame)
        np.testing.assert_array_equal(
            np.array(
                metrics_scores["accuracy_score*"].loc["train"]),
            np.array(
                [0.9]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score*"].loc["test"]),
            np.array([0.95]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["train"]),
            np.array([0.91]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["test"]),
            np.array([0.96]))
        np.testing.assert_array_equal(np.array(metrics_scores["f1_score"].columns),
                                      np.array(["ada-0"]))

    def test_multiple_monoview_classifiers(self):
        metrics = {"accuracy_score*": {}, "f1_score": {}}
        results = [MonoviewResult(view_index=0,
                                  classifier_name="ada",
                                  view_name="0",
                                  metrics_scores={"accuracy_score*": [0.9, 0.95],
                                                  "f1_score": [0.91, 0.96]},
                                  full_labels_pred="",
                                  classifier_config="",
                                  classifier="",
                                  n_features="",
                                  hps_duration=0,
                                  fit_duration=0,
                                  pred_duration=0,
                                  class_metric_scores={}),
                   MonoviewResult(view_index=0,
                                  classifier_name="dt",
                                  view_name="1",
                                  metrics_scores={"accuracy_score*": [0.8, 0.85],
                                                  "f1_score": [0.81, 0.86]},
                                  full_labels_pred="",
                                  classifier_config="",
                                  classifier="",
                                  n_features="",
                                  hps_duration=0,
                                  fit_duration=0,
                                  pred_duration=0,
                                  class_metric_scores={},)
                   ]
        metrics_scores, class_met = get_metrics_scores(metrics,
                                                       results, [])
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score*"], pd.DataFrame)
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score*"].loc["train"]),
            np.array([0.9, 0.8]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score*"].loc["test"]),
            np.array([0.95, 0.85]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["train"]),
            np.array([0.91, 0.81]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["test"]),
            np.array([0.96, 0.86]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].columns),
            np.array(["ada-0", "dt-1"]))

    def test_mutiview_result(self):
        metrics = {"accuracy_score*": {}, "f1_score": {}}
        results = [MultiviewResult("mv", "", {"accuracy_score*": [0.7, 0.75],
                                              "f1_score": [0.71, 0.76]}, "", 0, 0, 0, {}, ""),
                   MonoviewResult(view_index=0,
                                  classifier_name="dt",
                                  view_name="1",
                                  metrics_scores={"accuracy_score*": [0.8, 0.85],
                                                  "f1_score": [0.81, 0.86]},
                                  full_labels_pred="",
                                  classifier_config="",
                                  classifier="",
                                  n_features="",
                                  hps_duration=0,
                                  fit_duration=0,
                                  pred_duration=0,
                                  class_metric_scores={})
                   ]
        metrics_scores, class_met = get_metrics_scores(metrics,
                                                       results, [])
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score*"], pd.DataFrame)
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score*"].loc["train"]),
            np.array([0.7, 0.8]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score*"].loc["test"]),
            np.array([0.75, 0.85]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["train"]),
            np.array([0.71, 0.81]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["test"]),
            np.array([0.76, 0.86]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].columns),
            np.array(["mv", "dt-1"]))


class Test_init_plot(unittest.TestCase):

    def test_simple(self):
        results = []
        metric_name = "acc"
        data = np.random.RandomState(42).uniform(0, 1, (2, 2))
        metric_dataframe = pd.DataFrame(index=["train", "test"],
                                        columns=["dt-1", "mv"], data=data)
        directory = "dir"
        database_name = 'db'
        labels_names = ['lb1', "lb2"]
        class_met = metric_dataframe = pd.DataFrame(index=["train", "test"],
                                                    columns=["dt-1", "mv"], data=data)
        train, test, classifier_names, \
            file_name, nb_results, results, class_test = init_plot(results,
                                                                   metric_name,
                                                                   metric_dataframe,
                                                                   directory,
                                                                   database_name,
                                                                   class_met)
        self.assertEqual(file_name, os.path.join("dir", "db-acc"))
        np.testing.assert_array_equal(train, data[0, :])
        np.testing.assert_array_equal(test, data[1, :])
        np.testing.assert_array_equal(
            classifier_names, np.array(["dt-1", "mv"]))
        self.assertEqual(nb_results, 2)
        self.assertEqual(results, [["dt-1", "acc", data[1, 0], 0.0, data[1, 0]],
                                   ["mv", "acc", data[1, 1], 0.0, data[1, 1]]])


class Test_small_func(unittest.TestCase):

    def test_fig_size(self):
        kw, width = get_fig_size(5)
        self.assertEqual(kw, {"figsize": (15, 5)})
        self.assertEqual(width, 0.35)
        kw, width = get_fig_size(100)
        self.assertEqual(kw, {"figsize": (100, 100 / 3)})
        self.assertEqual(width, 0.35)

    def test_sort_by_test_scores(self):
        train_scores = np.array([1, 2, 3, 4])
        test_scores = np.array([4, 3, 2, 1])
        train_STDs = np.array([1, 2, 3, 4])
        test_STDs = np.array([1, 2, 3, 4])
        names = np.array(['1', '2', '3', '4'])
        sorted_names, sorted_train_scores, \
            sorted_test_scores, sorted_train_STDs, \
            sorted_test_STDs = sort_by_test_score(train_scores, test_scores,
                                                  names, train_STDs, test_STDs)
        np.testing.assert_array_equal(
            sorted_names, np.array(['4', '3', '2', '1']))
        np.testing.assert_array_equal(sorted_test_scores, [1, 2, 3, 4])
        np.testing.assert_array_equal(sorted_test_STDs, [4, 3, 2, 1])
        np.testing.assert_array_equal(sorted_train_scores, [4, 3, 2, 1])
        np.testing.assert_array_equal(sorted_train_STDs, [4, 3, 2, 1])
