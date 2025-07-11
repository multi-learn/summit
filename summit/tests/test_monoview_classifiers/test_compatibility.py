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
# import os
# import unittest
#

# Actuellement problématique a cause de la pep8isation du code. A voir
# plus tard


# import numpy as np
#
# from ...multiview_platform import monoview_classifiers
#
#
# class Test_methods(unittest.TestCase):
#
#     def test_simple(self):
#         for fileName in os.listdir(
#                 "summit/multiview_platform/monoview_classifiers"):
#             if fileName[-3:] == ".py" and fileName != "__init__.py":
#                 monoview_classifier_module = getattr(monoview_classifiers,
#                                                      fileName[:-3])
#                 self.assertIn("formatCmdArgs", dir(monoview_classifier_module),
#                               fileName[
#                               :-3] + " must have getKWARGS method implemented")
#                 self.assertIn("paramsToSet", dir(monoview_classifier_module),
#                               fileName[
#                               :-3] + " must have randomizedSearch method implemented")
#                 #test to be changed find name of class not same name of module
#                 # self.assertIn(fileName[:-3], dir(monoview_classifier_module),
#                 #              fileName[
#                 #              :-3] + " must have it's own class implemented")
#
#                 monoview_classifier_class = getattr(monoview_classifier_module,
#                                                     fileName[:-3])
#                 self.assertTrue(
#                     hasattr(monoview_classifier_class, "getInterpret"),
#                     fileName[:-3] + " class must have getInterpret implemented")
#                 self.assertTrue(
#                     hasattr(monoview_classifier_class, "canProbas", ),
#                     fileName[:-3] + " class must have canProbas implemented")
#                 monoview_classifier_instance = monoview_classifier_class()
#                 self.assertTrue(
#                     hasattr(monoview_classifier_instance, "param_names", ),
#                     fileName[:-3] + " class must have param_names attribute")
#                 self.assertTrue(
#                     hasattr(monoview_classifier_instance, "classed_params", ),
#                     fileName[:-3] + " class must have classed_params attribute")
#                 self.assertTrue(
#                     hasattr(monoview_classifier_instance, "distribs", ),
#                     fileName[:-3] + " class must have distribs attribute")
#                 self.assertTrue(
#                     hasattr(monoview_classifier_instance, "weird_strings", ),
#                     fileName[:-3] + " class must have weird_strings attribute")
#                 # check_estimator(monoview_classifier_instance)
#
#
# class Test_canProbas(unittest.TestCase):
#
#     def test_outputs(self):
#         for fileName in os.listdir(
#                 "summit/multiview_platform/monoview_classifiers"):
#             if fileName[-3:] == ".py" and fileName != "__init__.py":
#                 monoview_classifier_module = getattr(monoview_classifiers,
#                                                      fileName[:-3])
#                 monoview_classifier_class = getattr(monoview_classifier_module,
#                                                     fileName[:-3])()
#                 res = monoview_classifier_class.canProbas()
#                 self.assertEqual(type(res), bool,
#                                  "canProbas must return a boolean")
#
#     def test_inputs(self):
#         for fileName in os.listdir(
#                 "summit/multiview_platform/monoview_classifiers"):
#             if fileName[-3:] == ".py" and fileName != "__init__.py":
#                 monoview_classifier_module = getattr(monoview_classifiers,
#                                                      fileName[:-3])
#                 monoview_classifier_class = getattr(monoview_classifier_module,
#                                                     fileName[:-3])()
#                 with self.assertRaises(TypeError,
#                                        msg="canProbas must have 0 args") as catcher:
#                     monoview_classifier_class.canProbas(35)
#
#
# class Test_fit(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         cls.dataset = cls.random_state.random_sample((10, 20))
#         cls.labels = cls.random_state.randint(0, 2, 10)
#
#     # def test_inputs(cls):
#     #     # DATASET, CLASS_LABELS, random_state, NB_CORES=1, **kwargs
#     #     for fileName in os.listdir("Code/multiview_platform/monoview_classifiers"):
#     #         if fileName[-3:] == ".py" and fileName != "__init__.py":
#     #             monoview_classifier_module = getattr(monoview_classifiers, fileName[:-3])
#     #             cls.args = dict((str(index), value) for index, value in
#     #                             enumerate(monoview_classifier_module.paramsToSet(1, cls.random_state)[0]))
#     #             res = monoview_classifier_module.fit(cls.dataset, cls.labels, cls.random_state, **cls.args)
#     #             with cls.assertRaises(TypeError, msg="fit must have 3 positional args, one kwarg") as catcher:
#     #                 monoview_classifier_module.fit()
#     #                 monoview_classifier_module.fit(cls.dataset)
#     #                 monoview_classifier_module.fit(cls.dataset,cls.labels)
#     #                 monoview_classifier_module.fit(cls.dataset,cls.labels, cls.random_state, 1, 10)
#
#     # def test_outputs(cls):
#     #     for fileName in os.listdir("Code/multiview_platform/monoview_classifiers"):
#     #         if fileName[-3:] == ".py" and fileName != "__init__.py":
#     #             monoview_classifier_module = getattr(monoview_classifiers, fileName[:-3])
#     #             cls.args = dict((str(index), value) for index, value in
#     #                             enumerate(monoview_classifier_module.paramsToSet(1, cls.random_state)[0]))
#     #             res = monoview_classifier_module.fit(cls.dataset, cls.labels, cls.random_state, **cls.args)
#     #             cls.assertIn("predict", dir(res), "fit must return an object able to predict")
#
#
# class Test_paramsToSet(unittest.TestCase):
#
#     def test_inputs(self):
#         for fileName in os.listdir(
#                 "summit/multiview_platform/monoview_classifiers"):
#             if fileName[-3:] == ".py" and fileName != "__init__.py":
#                 monoview_classifier_module = getattr(monoview_classifiers,
#                                                      fileName[:-3])
#                 with self.assertRaises(TypeError,
#                                        msg="paramsToSet must have 2 positional args") as catcher:
#                     monoview_classifier_module.paramsToSet(2,
#                                                            np.random.RandomState(
#                                                                42), 10)
#                     monoview_classifier_module.paramsToSet(2)
#                     monoview_classifier_module.paramsToSet()
#                 res = monoview_classifier_module.paramsToSet(2,
#                                                              np.random.RandomState(
#                                                                  42))
#
#     def test_outputs(self):
#         for fileName in os.listdir(
#                 "summit/multiview_platform/monoview_classifiers"):
#             if fileName[-3:] == ".py" and fileName != "__init__.py":
#                 monoview_classifier_module = getattr(monoview_classifiers,
#                                                      fileName[:-3])
#                 res = monoview_classifier_module.paramsToSet(2,
#                                                              np.random.RandomState(
#                                                                  42))
#                 self.assertEqual(type(res), list)
#                 self.assertEqual(len(res), 2)
#                 self.assertEqual(type(res[0]), dict)
#
# # class Test_getKWARGS(unittest.TestCase):
# #
# #     # TODO : Find a way to enter the right args
# #
# #     def test_inputs(self):
# #         for fileName in os.listdir("Code/multiview_platform/monoview_classifiers"):
# #             if fileName[-3:] == ".py" and fileName != "__init__.py":
# #                 monoview_classifier_module = getattr(monoview_classifiers, fileName[:-3])
# #                 with self.assertRaises(TypeError, msg="getKWARGS must have 1 positional args") as catcher:
# #                     monoview_classifier_module.getKWARGS()
# #                     monoview_classifier_module.getKWARGS([1],2)
