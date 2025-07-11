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
import numpy as np


def get_names(classed_list):
    return np.array([object_.__class__.__name__ for object_ in classed_list])

# class BaseMultiviewClassifier(BaseEstimator, ClassifierMixin):
#
#     def __init__(self, random_state):
#         self.random_state = random_state
#
#     def genBestParams(self, detector):
#         return dict((param_name, detector.best_params_[param_name])
#                     for param_name in self.param_names)
#
#     def genParamsFromDetector(self, detector):
#         if self.classed_params:
#             classed_dict = dict((classed_param, get_names(
#                 detector.cv_results_["param_" + classed_param]))
#                                 for classed_param in self.classed_params)
#         if self.param_names:
#             return [(param_name,
#                      np.array(detector.cv_results_["param_" + param_name]))
#                     if param_name not in self.classed_params else (
#                 param_name, classed_dict[param_name])
#                     for param_name in self.param_names]
#         else:
#             return [()]
#
#     def genDistribs(self):
#         return dict((param_name, distrib) for param_name, distrib in
#                     zip(self.param_names, self.distribs))
#
#     def getConfig(self):
#         if self.param_names:
#             return "\n\t\t- " + self.__class__.__name__ + "with " + ", ".join(
#                 [param_name + " : " + self.to_str(param_name) for param_name in
#                  self.param_names])
#         else:
#             return "\n\t\t- " + self.__class__.__name__ + "with no config."
#
#     def to_str(self, param_name):
#         if param_name in self.weird_strings:
#             if self.weird_strings[param_name] == "class_name":
#                 return self.get_params()[param_name].__class__.__name__
#             else:
#                 return self.weird_strings[param_name](
#                     self.get_params()[param_name])
#         else:
#             return str(self.get_params()[param_name])
#
#     def get_interpretation(self):
#         return "No detailed interpretation function"

#
# def get_train_views_indices(dataset, train_indices, view_indices, ):
#     """This function  is used to get all the samples indices and view indices if needed"""
#     if view_indices is None:
#         view_indices = np.arange(dataset.nb_view)
#     if train_indices is None:
#         train_indices = range(dataset.get_nb_samples())
#     return train_indices, view_indices
