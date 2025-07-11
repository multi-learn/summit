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

from summit.multiview_platform.monoview.monoview_utils import MonoviewResult
from summit.multiview_platform.multiview.multiview_utils import MultiviewResult

from summit.multiview_platform.result_analysis.error_analysis import get_sample_errors, gen_error_data, gen_error_data_glob


class Test_get_sample_errors(unittest.TestCase):

    def test_simple(self):
        ground_truth = np.array([0, 1, 0, 1, 0, 1, 0, 1, -100])
        results = [MultiviewResult("mv", "", {"accuracy_score": [0.7, 0.75],
                                              "f1_score": [0.71, 0.76]},
                                   np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]),
                                   0, 0, 0, {}, "clf"),
                   MonoviewResult(0,
                                  "dt",
                                  "1",
                                  {"accuracy_score": [0.8, 0.85],
                                   "f1_score": [0.81, 0.86]}, np.array([0, 0, 1, 1, 0, 0, 1, 1, 0]), "", "",
                                  "", "", 0, 0, {})
                   ]
        sample_errors = get_sample_errors(ground_truth,
                                          results)
        self.assertIsInstance(sample_errors, dict)
        np.testing.assert_array_equal(sample_errors["mv"],
                                      np.array([1, 0, 1, 0, 0, 1, 0, 1, -100]))
        np.testing.assert_array_equal(sample_errors["dt-1"],
                                      np.array([1, 0, 0, 1, 1, 0, 0, 1, -100]))


class Test_gen_error_data(unittest.TestCase):

    def test_simple(self):
        random_state = np.random.RandomState(42)
        ada_data = random_state.randint(0, 2, size=7)
        mv_data = random_state.randint(0, 2, size=7)
        sample_errors = {"ada-1": ada_data,
                         "mv": mv_data}
        nb_classifiers, nb_samples, classifiers_names, \
            data_2d, error_on_samples = gen_error_data(sample_errors)
        self.assertEqual(nb_classifiers, 2)
        self.assertEqual(nb_samples, 7)
        self.assertEqual(classifiers_names, ["ada-1", "mv"])
        np.testing.assert_array_equal(
            data_2d, np.array([ada_data, mv_data]).transpose())
        np.testing.assert_array_equal(
            error_on_samples,
            (ada_data + mv_data) / nb_classifiers)


class Test_gen_error_data_glob(unittest.TestCase):

    def test_simple(self):
        random_state = np.random.RandomState(42)

        ada_error_data_1 = random_state.randint(0, 2, 7)
        ada_error_data_2 = random_state.randint(0, 2, 7)
        ada_sum = ada_error_data_1 + ada_error_data_2
        mv_error_data_1 = random_state.randint(0, 2, 7)
        mv_error_data_2 = random_state.randint(0, 2, 7)
        mv_sum = mv_error_data_1 + mv_error_data_2

        combi_results = {"ada-1": ada_sum, "mv": mv_sum}

        stats_iter = 2

        nb_samples, nb_classifiers, \
            data, error_on_samples, \
            classifier_names = gen_error_data_glob(combi_results,
                                                   stats_iter)
        self.assertEqual(nb_samples, 7)
        self.assertEqual(nb_classifiers, 2)
        np.testing.assert_array_equal(
            data, np.array([ada_sum, mv_sum]).transpose())
        np.testing.assert_array_equal(error_on_samples, np.sum(
            np.array([ada_sum, mv_sum]), axis=0) / (nb_classifiers * stats_iter))
        self.assertEqual(classifier_names, ["ada-1", "mv"])
