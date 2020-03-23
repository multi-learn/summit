import unittest
import numpy as np
import pandas as pd
import os

from multiview_platform.mono_multi_view_classifiers.monoview.monoview_utils import MonoviewResult
from multiview_platform.mono_multi_view_classifiers.multiview.multiview_utils import MultiviewResult

from multiview_platform.mono_multi_view_classifiers.result_analysis.metric_analysis import get_metrics_scores, init_plot

class Test_get_metrics_scores(unittest.TestCase):


    def test_simple(self):
        metrics = {"accuracy_score*":{},"f1_score":{}}
        results = [MonoviewResult(0,
                                  "ada",
                                  "0",
                                  {"accuracy_score*":[0.9, 0.95],
                                   "f1_score":[0.91, 0.96]}
                                  , "", "", "", "", "",0,0,{})]
        metrics_scores, class_met = get_metrics_scores(metrics,
                                                            results, [])
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score*"], pd.DataFrame)
        np.testing.assert_array_equal(np.array(metrics_scores["accuracy_score*"].loc["train"]), np.array([0.9]))
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
        metrics = {"accuracy_score*":{},"f1_score":{}}
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
                                  class_metric_scores={})
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
        metrics = {"accuracy_score*":{},"f1_score":{}}
        results = [MultiviewResult("mv", "", {"accuracy_score*": [0.7, 0.75],
                                   "f1_score": [0.71, 0.76]}, "",0,0,0, {}),
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
        data = np.random.RandomState(42).uniform(0,1,(2,2))
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
        np.testing.assert_array_equal(train, data[0,:])
        np.testing.assert_array_equal(test, data[1, :])
        np.testing.assert_array_equal(classifier_names, np.array(["dt-1", "mv"]))
        self.assertEqual(nb_results, 2)
        self.assertEqual(results, [["dt-1", "acc", data[1,0], 0.0, data[1,0]],
                                   ["mv", "acc", data[1,1], 0.0, data[1,1]]])