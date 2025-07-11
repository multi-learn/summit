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

from .fusion_utils import BaseFusionClassifier
from ...multiview.multiview_utils import BaseMultiviewClassifier, \
    get_available_monoview_classifiers, ConfigGenerator
from ...utils.dataset import get_samples_views_indices


class ClassifierDistribution:

    def __init__(self, seed=42, available_classifiers=None):
        self.random_state = np.random.RandomState(seed)
        self.available_classifiers = available_classifiers

    def draw(self, nb_view, rs=None):
        if rs is not None:
            self.random_state.seed(rs)
        return self.random_state.choice(self.available_classifiers,
                                        size=nb_view, replace=True)


class ClassifierCombinator:

    def __init__(self, need_probas=False):
        self.available_classifiers = get_available_monoview_classifiers(
            need_probas)

    def rvs(self, random_state=None):
        return ClassifierDistribution(seed=random_state.randint(1),
                                      available_classifiers=self.available_classifiers)


class ConfigDistribution:

    def __init__(self, seed=42, available_classifiers=None):
        self.random_state = np.random.RandomState(seed)
        self.config_generator = ConfigGenerator(available_classifiers)

    def draw(self, nb_view, rs=None):
        if rs is not None:
            self.random_state.seed(rs)
        config_samples = [self.config_generator.rvs(self.random_state)
                          for _ in range(nb_view)]
        return config_samples


class MultipleConfigGenerator:

    def __init__(self, ):
        self.available_classifiers = get_available_monoview_classifiers()

    def rvs(self, random_state=None):
        return ConfigDistribution(seed=random_state.randint(1),
                                  available_classifiers=self.available_classifiers)


class WeightDistribution:

    def __init__(self, seed=42, distribution_type="uniform"):
        self.random_state = np.random.RandomState(seed)
        self.distribution_type = distribution_type

    def draw(self, nb_view):
        if self.distribution_type == "uniform":
            return self.random_state.random_sample(nb_view)


class WeightsGenerator:

    def __init__(self, distibution_type="uniform"):
        self.distribution_type = distibution_type

    def rvs(self, random_state=None):
        return WeightDistribution(seed=random_state.randint(1),
                                  distribution_type=self.distribution_type)


class LateFusionClassifier(BaseMultiviewClassifier, BaseFusionClassifier):

    def __init__(self, random_state=None, classifiers_names=None,
                 classifier_configs=None, nb_cores=1, weights=None,
                 rs=None):
        BaseMultiviewClassifier.__init__(self, random_state)
        self.classifiers_names = classifiers_names
        self.classifier_configs = classifier_configs
        self.nb_cores = nb_cores
        self.weights = weights
        self.rs = rs
        self.param_names = ["classifiers_names", "classifier_configs",
                            "weights", "rs"]
        self.distribs = [ClassifierCombinator(need_probas=self.need_probas),
                         MultipleConfigGenerator(),
                         WeightsGenerator(),
                         np.arange(1000)]

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_samples_views_indices(X,
                                                                train_indices,
                                                                view_indices)
        self.used_views = view_indices
        if np.unique(y).shape[0] > 2:
            multiclass = True
        else:
            multiclass = False
        self.init_params(len(view_indices), multiclass)
        if np.unique(y[train_indices]).shape[0] > 2:
            raise ValueError("Multiclass not supported")
        self.monoview_estimators = [
            monoview_estimator.fit(X.get_v(view_index, train_indices),
                                   y[train_indices])
            for view_index, monoview_estimator
            in zip(view_indices,
                   self.monoview_estimators)]
        return self

    def init_params(self, nb_view, mutliclass=False):
        if self.weights is None:
            self.weights = np.ones(nb_view) / nb_view
        elif isinstance(self.weights, WeightDistribution):
            self.weights = self.weights.draw(nb_view)
        else:
            self.weights = self.weights / np.sum(self.weights)

        self.init_classifiers(nb_view)

        self.monoview_estimators = [
            self.init_monoview_estimator(classifier_name,
                                         self.classifier_configs[
                                             classifier_index],
                                         classifier_index=classifier_index,
                                         multiclass=mutliclass)
            for classifier_index, classifier_name
            in enumerate(self.classifiers_names)]

    def init_classifiers(self, nb_view, nb_monoview_per_view=None):
        if nb_monoview_per_view is not None:
            nb_clfs = nb_monoview_per_view
        else:
            nb_clfs = nb_view

        if isinstance(self.classifiers_names, ClassifierDistribution):
            self.classifiers_names = self.classifiers_names.draw(nb_clfs,
                                                                 self.rs)
        elif self.classifiers_names is None:
            self.classifiers_names = ["decision_tree" for _ in range(nb_clfs)]
        elif isinstance(self.classifiers_names, str):
            self.classifiers_names = [self.classifiers_names
                                      for _ in range(nb_clfs)]

        if isinstance(self.classifier_configs, ConfigDistribution):
            self.classifier_configs = [
                {classifier_name: config[classifier_name]} for
                config, classifier_name in
                zip(self.classifier_configs.draw(nb_clfs,
                                                 self.rs),
                    self.classifiers_names)]
        elif isinstance(self.classifier_configs, dict):
            self.classifier_configs = [
                {classifier_name: self.classifier_configs[classifier_name]} for
                classifier_name in self.classifiers_names]
        elif self.classifier_configs is None:
            self.classifier_configs = [None for _ in range(nb_clfs)]

    # def verif_clf_views(self, classifier_names, nb_view):
    #     if classifier_names is None:
    #         if nb_view is None:
    #             raise AttributeError(self.__class__.__name__+" must have either classifier_names or nb_views provided.")
    #         else:
    #             self.classifiers_names = self.get_classifiers(get_available_monoview_classifiers(), nb_view)
    #     else:
    #         if nb_view is None:
    #             self.classifiers_names = classifier_names
    #         else:
    #             if len(classifier_names)==nb_view:
    #                 self.classifiers_names = classifier_names
    #             else:
    #                 warnings.warn("nb_view and classifier_names not matching, choosing nb_view random classifiers in classifier_names.", UserWarning)
    #                 self.classifiers_names = self.get_classifiers(classifier_names, nb_view)

    def get_classifiers(self, classifiers_names, nb_choices):
        return self.random_state.choice(classifiers_names, size=nb_choices,
                                        replace=True)
