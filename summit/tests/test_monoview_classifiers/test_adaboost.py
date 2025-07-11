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
# import unittest
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
#
# from ...multiview_platform.monoview_classifiers import Adaboost
#
#
# class Test_canProbas(unittest.TestCase):
#
#     def test_simple(cls):
#         cls.assertTrue(Adaboost.canProbas())
#
#
# class Test_paramsToSet(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.n_iter = 4
#         cls.random_state = np.random.RandomState(42)
#
#     def test_simple(cls):
#         res = Adaboost.paramsToSet(cls.n_iter, cls.random_state)
#         cls.assertEqual(len(res), cls.n_iter)
#         cls.assertEqual(type(res[0][0]), int)
#         cls.assertEqual(type(res[0][1]), type(DecisionTreeClassifier()))
#         cls.assertEqual([7,4,13,11], [resIter[0] for resIter in res])
#
#
# class Test_getKWARGS(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.kwargs_list = [("CL_Adaboost_n_est", 10),
#                            ("CL_Adaboost_b_est", DecisionTreeClassifier())]
#
#     def test_simple(cls):
#         res = Adaboost.getKWARGS(cls.kwargs_list)
#         cls.assertIn("0", res)
#         cls.assertIn("1", res)
#         cls.assertEqual(type(res), dict)
#         cls.assertEqual(res["0"], 10)
#         # Can't test decision tree
#
#     def test_wrong(cls):
#         cls.kwargs_list[0] = ("chicken_is_heaven",42)
#         with cls.assertRaises(ValueError) as catcher:
#             Adaboost.getKWARGS(cls.kwargs_list)
#         exception = catcher.exception
#         # cls.assertEqual(exception, "Wrong arguments served to Adaboost")
#
#
# class Test_randomizedSearch(unittest.TestCase):
#
#     def test_simple(cls):
#         pass  # Test with simple params
#
#
# class Test_fit(unittest.TestCase):
#
#     def setUp(self):
#         self.random_state = np.random.RandomState(42)
#         self.dataset = self.random_state.randint(0, 100, (10, 5))
#         self.labels = self.random_state.randint(0, 2, 10)
#         self.kwargs = {"0": 5}
#         self.classifier = Adaboost.fit(self.dataset, self.labels, 42, NB_CORES=1, **self.kwargs)
#
#     def test_fit_kwargs_string(self):
#         self.kwargs = {"0": "5"}
#         classifier = Adaboost.fit(self.dataset, self.labels, 42, NB_CORES=1, **self.kwargs)
#         self.assertEqual(classifier.n_estimators, 5)
#
#     def test_fit_kwargs_int(self):
#         self.kwargs = {"0": 5}
#         classifier = Adaboost.fit(self.dataset, self.labels, 42, NB_CORES=1, **self.kwargs)
#         self.assertEqual(classifier.n_estimators, 5)
#
#     def test_fit_labels(self):
#         predicted_labels = self.classifier.predict(self.dataset)
#         np.testing.assert_array_equal(predicted_labels, self.labels)
#
