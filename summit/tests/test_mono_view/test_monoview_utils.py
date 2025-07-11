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
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from summit.multiview_platform.monoview import monoview_utils
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint


class TestFunctions(unittest.TestCase):

    def test_gen_test_folds_preds(self):
        self.random_state = np.random.RandomState(42)
        self.X_train = self.random_state.random_sample((31, 10))
        self.y_train = np.ones(31, dtype=int)
        self.KFolds = StratifiedKFold(n_splits=3, )

        self.estimator = DecisionTreeClassifier(max_depth=1)

        self.y_train[15:] = -1
        testFoldsPreds = monoview_utils.gen_test_folds_preds(self.X_train,
                                                             self.y_train,
                                                             self.KFolds,
                                                             self.estimator)
        self.assertEqual(testFoldsPreds.shape, (3, 10))
        np.testing.assert_array_equal(testFoldsPreds[0], np.array(
            [1, 1, -1, -1, 1, 1, -1, 1, -1, 1]))

    def test_change_label_to_minus(self):
        lab = monoview_utils.change_label_to_minus(np.array([0, 1, 0]))
        np.testing.assert_array_equal(lab, np.array([-1, 1, -1]))

    def test_change_label_to_zero(self):
        lab = monoview_utils.change_label_to_zero(np.array([-1, 1, -1]))
        np.testing.assert_array_equal(lab, np.array([0, 1, 0]))

    def test_compute_possible_combinations(self):
        n_possib = monoview_utils.compute_possible_combinations(
            {"a": [1, 2], "b": {"c": [2, 3]}, "d": CustomRandint(0, 10)})
        np.testing.assert_array_equal(n_possib, np.array([2, np.inf, 10]))


class FakeClf(monoview_utils.BaseMonoviewClassifier):

    def __init__(self):
        pass


class TestBaseMonoviewClassifier(unittest.TestCase):

    def test_simple(self):
        name = FakeClf().get_name_for_fusion()
        self.assertEqual(name, 'Fake')
