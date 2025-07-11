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

from summit.multiview_platform.result_analysis.execution import format_previous_results, get_arguments, analyze_iterations
from summit.tests.utils import rm_tmp, tmp_path, test_dataset

class FakeClf():

    def __init__(self):
        self.feature_importances_ = [0.01,0.99]


class FakeClassifierResult:

    def __init__(self, i=1):
        self.classifier_name = 'test' + str(i)
        self.full_labels_pred = np.array([0, 1, 1, 2, 1])
        self.hps_duration = i
        self.fit_duration = i
        self.pred_duration = i
        self.clf=FakeClf()

    def get_classifier_name(self):
        return self.classifier_name


class Test_format_previous_results(unittest.TestCase):

    def test_simple(self):
        iter_results = {
            "metrics_scores": [],
            "sample_errors": [],
            "feature_importances": [],
            "labels": [],
            "durations": [],
            "class_metrics_scores": []}
        random_state = np.random.RandomState(42)

        # Gen metrics data
        metrics_1_data = random_state.uniform(size=(2, 2))
        metrics_2_data = random_state.uniform(size=(2, 2))
        metric_1_df = pd.DataFrame(data=metrics_1_data, index=["train", "test"],
                                   columns=["ada-1", "mv"])
        metric_2_df = pd.DataFrame(data=metrics_2_data, index=["train", "test"],
                                   columns=["ada-1", "mv"])
        iter_results["metrics_scores"].append({"acc": metric_1_df})
        iter_results["metrics_scores"].append({"acc": metric_2_df})

        # Gen error data
        ada_error_data_1 = random_state.randint(0, 2, 7)
        ada_error_data_2 = random_state.randint(0, 2, 7)
        ada_sum = ada_error_data_1 + ada_error_data_2
        mv_error_data_1 = random_state.randint(0, 2, 7)
        mv_error_data_2 = random_state.randint(0, 2, 7)
        mv_sum = mv_error_data_1 + mv_error_data_2
        iter_results["sample_errors"].append({})
        iter_results["sample_errors"].append({})
        iter_results["sample_errors"][0]["ada-1"] = ada_error_data_1
        iter_results["sample_errors"][0]["mv"] = mv_error_data_1
        iter_results["sample_errors"][1]["ada-1"] = ada_error_data_2
        iter_results["sample_errors"][1]["mv"] = mv_error_data_2

        iter_results["durations"].append(pd.DataFrame(index=["ada-1", "mv"],
                                                      columns=["plif", "plaf"],
                                                      data=np.zeros((2, 2))))
        iter_results["durations"].append(pd.DataFrame(index=["ada-1", "mv"],
                                                      columns=["plif",
                                                               "plaf"],
                                                      data=np.ones((2, 2))))

        # Running the function
        metric_analysis, class_met, error_analysis, \
            feature_importances, feature_stds, \
            labels, durations_mean, duration_std = format_previous_results(
                iter_results)
        mean_df = pd.DataFrame(data=np.mean(np.array([metrics_1_data,
                                                      metrics_2_data]),
                                            axis=0),
                               index=["train", "test"],
                               columns=["ada-1", "mvm"])
        std_df = pd.DataFrame(data=np.std(np.array([metrics_1_data,
                                                    metrics_2_data]),
                                          axis=0),
                              index=["train", "test"],
                              columns=["ada-1", "mvm"])

        # Testing
        np.testing.assert_array_almost_equal(metric_analysis["acc"]["mean"].loc["train"],
                                      mean_df.loc["train"])
        np.testing.assert_array_almost_equal(metric_analysis["acc"]["mean"].loc["test"],
                                      mean_df.loc["test"])
        np.testing.assert_array_almost_equal(metric_analysis["acc"]["std"].loc["train"],
                                      std_df.loc["train"])
        np.testing.assert_array_almost_equal(metric_analysis["acc"]["std"].loc["test"],
                                      std_df.loc["test"])
        np.testing.assert_array_almost_equal(ada_sum, error_analysis["ada-1"])
        np.testing.assert_array_almost_equal(mv_sum, error_analysis["mv"])
        self.assertEqual(durations_mean.at["ada-1", 'plif'], 0.5)


class Test_get_arguments(unittest.TestCase):

    def setUp(self):
        self.benchamrk_argument_dictionaries = [{"flag": "good_flag", "valid": True},
                                                {"flag": "bad_flag", "valid": False}]

    def test_benchmark_wanted(self):
        argument_dict = get_arguments(
            self.benchamrk_argument_dictionaries, "good_flag")
        self.assertTrue(argument_dict["valid"])


class Test_analyze_iterations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.results = [[0, [FakeClassifierResult(), FakeClassifierResult(i=2)], []], [
            1, [FakeClassifierResult(), FakeClassifierResult(i=2)], []]]
        cls.benchmark_argument_dictionaries = [
            {
                "labels_dictionary": {
                    0: "zero", 1: "one", 2: "two"}, "flag": 0, "directory": tmp_path, "args": {
                    "name": "test_dataset"}}, {
                "labels_dictionary": {
                    0: "zero", 1: "one", 2: "two"}, "flag": 1, "directory": tmp_path, "args": {
                    "name": "test_dataset"}}]
        cls.stats_iter = 2
        cls.metrics = {}
        cls.sample_ids = ['ex1', 'ex5', 'ex4', 'ex3', 'ex2', ]
        cls.labels = np.array([0, 1, 2, 1, 1])
        cls.feature_ids = [["a", "b"]]
        cls.view_names = [""]

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        analysis = analyze_iterations(self.results,
                                      self.benchmark_argument_dictionaries,
                                      self.stats_iter,
                                      self.metrics,
                                      self.sample_ids,
                                      self.labels,
                                      self.feature_ids,
                                      self.view_names)
        res, iter_res, tracebacks, labels_names = analysis
        self.assertEqual(labels_names, ['zero', 'one', 'two'])

        self.assertEqual(iter_res['class_metrics_scores'], [{}, {}])

        pd.testing.assert_frame_equal(iter_res['durations'][0], pd.DataFrame(index=['test1', 'test2'],
                                                                             columns=[
                                                                                 'hps', 'fit', 'pred'],
                                                                             data=np.array([1, 1, 1, 2, 2, 2]).reshape((2, 3)), dtype=object))
        np.testing.assert_array_equal(
            iter_res['sample_errors'][0]['test1'], np.array([1, 1, 0, 0, 1]))
        np.testing.assert_array_equal(
            iter_res['labels'], np.array([0, 1, 2, 1, 1]))
        self.assertEqual(iter_res['metrics_scores'], [{}, {}])
