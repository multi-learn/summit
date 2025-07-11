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
from .. import multiview_classifiers
from abc import abstractmethod

import numpy as np

from .. import monoview_classifiers
from ..utils.base import BaseClassifier, ResultAnalyser
from ..utils.dataset import RAMDataset

class BaseMultiviewClassifier(BaseClassifier):
    """
    BaseMultiviewClassifier base of Multiview classifiers

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number multiview_generator to use when
        shuffling the data.
    """

    def __init__(self, random_state):

        self.random_state = random_state
        self.short_name = self.__module__.split(".")[-1]
        self.weird_strings = {}
        self.used_views = None

    @abstractmethod
    def fit(self, X, y, train_indices=None,
            view_indices=None):  # pragma: no cover
        pass

    @abstractmethod
    def predict(self, X, sample_indices=None,
                view_indices=None):  # pragma: no cover
        pass

    def _check_views(self, view_indices):  # pragma: no cover
        if self.used_views is not None and not np.array_equal(
                np.sort(self.used_views), np.sort(view_indices)):
            raise ValueError(
                'Used {} views to fit, and trying to predict on {}'.format(
                    self.used_views, view_indices))

    def accepts_multi_class(self, random_state, n_samples=10, dim=2,
                            n_classes=3, n_views=2):
        if int(n_samples / n_classes) < 1:
            raise ValueError(
                "n_samples ({}) / n_classes ({}) must be over 1".format(
                    n_samples,
                    n_classes))
        fake_mc_X = RAMDataset(
            views=[random_state.randint(low=0, high=101,
                                        size=(n_samples, dim))
                   for i in range(n_views)],
            labels=[class_index
                    for _ in range(int(n_samples / n_classes))
                    for class_index in range(n_classes)],
            are_sparse=False,
            name="mc_dset",
            labels_names=[str(class_index)
                          for class_index in range(n_classes)],
            view_names=["V0", "V1"],
        )

        fake_mc_y = [class_index
                     for _ in range(int(n_samples / n_classes))
                     for class_index in range(n_classes)]
        fake_mc_y += [0 for _ in range(n_samples % n_classes)]
        fake_mc_y = np.asarray(fake_mc_y)
        try:
            self.fit(fake_mc_X, fake_mc_y)
            return True
        except ValueError:
            return False


class ConfigGenerator():

    def __init__(self, classifier_names):
        self.distribs = {}
        for classifier_name in classifier_names:
            classifier_class = get_monoview_classifier(classifier_name)
            self.distribs[classifier_name] = dict((param_name, param_distrib)
                                                  for param_name, param_distrib
                                                  in
                                                  zip(
                                                      classifier_class().param_names,
                                                      classifier_class().distribs)
                                                  if
                                                  param_name != "random_state")

    def rvs(self, random_state=None):
        config_sample = {}
        for classifier_name, classifier_config in self.distribs.items():
            config_sample[classifier_name] = {}
            for param_name, param_distrib in classifier_config.items():
                if hasattr(param_distrib, "rvs"):
                    config_sample[classifier_name][
                        param_name] = param_distrib.rvs(
                        random_state=random_state)
                else:
                    config_sample[classifier_name][
                        param_name] = param_distrib[
                        random_state.randint(len(param_distrib))]
        return config_sample


def get_available_monoview_classifiers(need_probas=False):
    available_classifiers = [module_name
                             for module_name in dir(monoview_classifiers)
                             if not (
                                 module_name.startswith("__") or module_name == "additions")]
    if need_probas:
        proba_classifiers = []
        for module_name in available_classifiers:
            module = getattr(monoview_classifiers, module_name)
            classifier_class = getattr(module, module.classifier_class_name)()
            proba_prediction = getattr(classifier_class, "predict_proba", None)
            if callable(proba_prediction):
                proba_classifiers.append(module_name)
        available_classifiers = proba_classifiers
    return available_classifiers


def get_monoview_classifier(classifier_name, multiclass=False):
    classifier_module = getattr(monoview_classifiers, classifier_name)
    classifier_class = getattr(classifier_module,
                               classifier_module.classifier_class_name)
    return classifier_class


class MultiviewResult(object):
    def __init__(self, classifier_name, classifier_config,
                 metrics_scores, full_labels, hps_duration, fit_duration,
                 pred_duration, class_metric_scores, clf):
        self.classifier_name = classifier_name
        self.classifier_config = classifier_config
        self.metrics_scores = metrics_scores
        self.full_labels_pred = full_labels
        self.hps_duration = hps_duration
        self.fit_duration = fit_duration
        self.pred_duration = pred_duration
        self.class_metric_scores = class_metric_scores
        self.clf=clf

    def get_classifier_name(self):
        try:
            multiview_classifier_module = getattr(multiview_classifiers,
                                                  self.classifier_name)
            multiview_classifier = getattr(multiview_classifier_module,
                                           multiview_classifier_module.classifier_class_name)(
                42, **self.classifier_config)
            return multiview_classifier.short_name
        except BaseException:
            return self.classifier_name


class MultiviewResultAnalyzer(ResultAnalyser):

    def __init__(self, view_names, classifier, classification_indices, k_folds,
                 hps_method, metrics_dict, n_iter, class_label_names,
                 pred, directory, base_file_name, labels,
                 database_name, nb_cores, duration, feature_ids):
        if hps_method.endswith("equiv"):
            n_iter = n_iter * len(view_names)
        ResultAnalyser.__init__(self, classifier, classification_indices,
                                k_folds,
                                hps_method, metrics_dict, n_iter,
                                class_label_names,
                                pred, directory,
                                base_file_name, labels, database_name,
                                nb_cores, duration, feature_ids)
        self.classifier_name = classifier.short_name
        self.view_names = view_names

    def get_base_string(self, ):
        return "Multiview classification on {}  with {}\n\n".format(
            self.database_name,
            self.classifier_name)

    def get_view_specific_info(self):
        return "\t- Views : " + ', '.join(self.view_names) + "\n"
