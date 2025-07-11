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
from sklearn.base import BaseEstimator

from summit.multiview_platform.utils.multiclass import get_mc_estim, \
    OVRWrapper, OVOWrapper, MultiviewOVOWrapper, MultiviewOVRWrapper


class FakeMCEstim(BaseEstimator):

    def __init__(self):
        self.short_name = "short_name"

    def accepts_multi_class(self, random_state):
        return False


class FakeEstimNative(FakeMCEstim):

    def accepts_multi_class(self, random_state):
        return True


class FakeNonProbaEstim(FakeMCEstim):
    pass


class FakeProbaEstim(FakeMCEstim):

    def predict_proba(self):
        pass


class Test_get_mc_estim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.y = cls.random_state.randint(0, 3, 10)

    def test_biclass(self):
        y = self.random_state.randint(0, 2, 10)
        estimator = "Test"
        returned_estimator = get_mc_estim(estimator, self.random_state, y=y)
        self.assertEqual(returned_estimator, estimator)

    def test_multiclass_native(self):
        estimator = FakeEstimNative()
        returned_estimator = get_mc_estim(
            estimator, self.random_state, y=self.y)
        self.assertIsInstance(returned_estimator, FakeEstimNative)

    def test_multiclass_ovo(self):
        estimator = FakeNonProbaEstim()
        returned_estimator = get_mc_estim(
            estimator, self.random_state, y=self.y)
        self.assertIsInstance(returned_estimator, OVOWrapper)

    def test_multiclass_ovr(self):
        estimator = FakeProbaEstim()
        returned_estimator = get_mc_estim(
            estimator, self.random_state, y=self.y)
        self.assertIsInstance(returned_estimator, OVRWrapper)

    def test_multiclass_ovo_multiview(self):
        estimator = FakeNonProbaEstim()
        returned_estimator = get_mc_estim(estimator, self.random_state,
                                          multiview=True, y=self.y, )
        self.assertIsInstance(returned_estimator, MultiviewOVOWrapper)

    def test_multiclass_ovr_multiview(self):
        estimator = FakeProbaEstim()
        returned_estimator = get_mc_estim(estimator, self.random_state,
                                          multiview=True, y=self.y,)
        self.assertIsInstance(returned_estimator, MultiviewOVRWrapper)


class FakeMVClassifier(BaseEstimator):

    def __init__(self, short_name="None"):
        self.short_name = short_name

    def fit(self, X, y, train_indices=None, view_indices=None):
        self.n_classes = np.unique(y[train_indices]).shape[0]
        self.views_indices = view_indices

    def predict(self, X, sample_indices=None, view_indices=None):
        self.sample_indices = sample_indices
        self.views_indices = view_indices
        return np.zeros((sample_indices.shape[0]))


class FakeMVClassifierProb(FakeMVClassifier):

    def predict_proba(self, X, sample_indices=None, view_indices=None):
        self.sample_indices = sample_indices
        self.views_indices = view_indices
        return np.zeros((sample_indices.shape[0], 2))


class Test_MultiviewOVRWrapper_fit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.X = "dataset"
        cls.n_classes = 3
        cls.y = cls.random_state.randint(0, cls.n_classes, 50)
        cls.train_indices = np.arange(25)
        cls.sample_indices = np.arange(25) + 25
        cls.view_indices = "None"
        cls.wrapper = MultiviewOVRWrapper(FakeMVClassifierProb(), )

    def test_fit(self):
        fitted = self.wrapper.fit(self.X, self.y, train_indices=self.train_indices,
                                  view_indices=self.view_indices)
        for estimator in fitted.estimators_:
            self.assertEqual(estimator.n_classes, 2)
            self.assertEqual(estimator.views_indices, "None")

    def test_predict(self):
        fitted = self.wrapper.fit(self.X, self.y, train_indices=self.train_indices,
                                  view_indices=self.view_indices)
        pred = fitted.predict(self.X, sample_indices=self.sample_indices,
                              view_indices=self.view_indices)
        for estimator in fitted.estimators_:
            np.testing.assert_array_equal(estimator.sample_indices,
                                          self.sample_indices)


class FakeDset:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def get_nb_samples(self):
        return self.n_samples


class Test_MultiviewOVOWrapper_fit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.n_samples = 50
        cls.X = FakeDset(n_samples=cls.n_samples)
        cls.n_classes = 3
        cls.y = cls.random_state.randint(0, cls.n_classes, cls.n_samples)
        cls.train_indices = np.arange(int(cls.n_samples / 2))
        cls.sample_indices = np.arange(
            int(cls.n_samples / 2)) + int(cls.n_samples / 2)
        cls.view_indices = "None"
        cls.wrapper = MultiviewOVOWrapper(FakeMVClassifier(), )

    def test_fit(self):
        fitted = self.wrapper.fit(self.X, self.y, train_indices=self.train_indices,
                                  view_indices=self.view_indices)
        for estimator in fitted.estimators_:
            self.assertEqual(estimator.n_classes, 2)
            self.assertEqual(estimator.views_indices, "None")

    def test_predict(self):
        fitted = self.wrapper.fit(self.X, self.y, train_indices=self.train_indices,
                                  view_indices=self.view_indices)
        pred = fitted.predict(self.X, sample_indices=self.sample_indices,
                              view_indices=self.view_indices)
        for estimator in fitted.estimators_:
            np.testing.assert_array_equal(estimator.sample_indices,
                                          self.sample_indices)


if __name__ == '__main__':
    unittest.main()
