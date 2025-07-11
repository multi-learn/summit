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
import os

from summit.tests.utils import rm_tmp, tmp_path, test_dataset

from summit.multiview_platform.multiview_classifiers import \
    weighted_linear_early_fusion


class Test_WeightedLinearEarlyFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.random_state = np.random.RandomState(42)
        cls.view_weights = [0.5, 0.5]
        cls.monoview_classifier_name = "decision_tree"
        cls.monoview_classifier_config = {
            "max_depth": 1, "criterion": "gini", "splitter": "best"}
        cls.classifier = weighted_linear_early_fusion.WeightedLinearEarlyFusion(
            random_state=cls.random_state, view_weights=cls.view_weights,
            monoview_classifier_name=cls.monoview_classifier_name,
            monoview_classifier_config=cls.monoview_classifier_config)
        cls.dataset = test_dataset

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        np.testing.assert_array_equal(
            self.view_weights, self.classifier.view_weights)

    def test_fit(self):
        self.assertRaises(AttributeError, getattr,
                          self.classifier.monoview_classifier, "classes_")
        self.classifier.fit(
            self.dataset,
            test_dataset.get_labels(),
            None,
            None)
        np.testing.assert_array_equal(self.classifier.monoview_classifier.classes_,
                                      np.array([0, 1]))

    def test_predict(self):
        self.classifier.fit(
            self.dataset,
            test_dataset.get_labels(),
            None,
            None)
        predicted_labels = self.classifier.predict(self.dataset, None, None)
        np.testing.assert_array_equal(
            predicted_labels, test_dataset.get_labels())

    def test_transform_data_to_monoview_simple(self):
        sample_indices, X = self.classifier.transform_data_to_monoview(self.dataset,
                                                                       None, None)
        self.assertEqual(X.shape, (5, 12))
        np.testing.assert_array_equal(X, np.concatenate(
            (self.dataset.get_v(0), self.dataset.get_v(1)), axis=1))
        np.testing.assert_array_equal(sample_indices, np.arange(5))

    def test_transform_data_to_monoview_view_select(self):
        sample_indices, X = self.classifier.transform_data_to_monoview(
            self.dataset,
            None, np.array([0]))
        self.assertEqual(X.shape, (5, 6))
        np.testing.assert_array_equal(X, self.dataset.get_v(0))
        np.testing.assert_array_equal(sample_indices, np.arange(5))

    def test_transform_data_to_monoview_sample_view_select(self):
        sample_indices, X = self.classifier.transform_data_to_monoview(
            self.dataset,
            np.array([1, 2, 3]), np.array([0]))
        self.assertEqual(X.shape, (3, 6))
        np.testing.assert_array_equal(X, self.dataset.get_v(0)[
                                      np.array([1, 2, 3]), :])
        np.testing.assert_array_equal(sample_indices, np.array([1, 2, 3]))
