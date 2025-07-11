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
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ..utils.base import BaseClassifier, ResultAnalyser
from ..utils.hyper_parameter_search import CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def change_label_to_minus(y):
    """
    Change the label 0 to minus one

    Parameters
    ----------
    y :

    Returns
    -------
    label y with -1 instead of 0

    """
    minus_y = np.copy(y)
    minus_y[np.where(y == 0)] = -1
    return minus_y


def change_label_to_zero(y):
    """
    Change the label -1 to 0

    Parameters
    ----------
    y

    Returns
    -------

    """
    zeroed_y = np.copy(y)
    zeroed_y[np.where(y == -1)] = 0
    return zeroed_y


def compute_possible_combinations(params_dict):
    n_possibs = np.ones(len(params_dict)) * np.inf
    for value_index, value in enumerate(params_dict.values()):
        if isinstance(value, list):
            n_possibs[value_index] = len(value)
        elif isinstance(value, CustomRandint):
            n_possibs[value_index] = value.get_nb_possibilities()
    return n_possibs


def gen_test_folds_preds(X_train, y_train, KFolds, estimator):
    test_folds_preds = []
    train_index = np.arange(len(y_train))
    folds = KFolds.split(train_index, y_train)
    fold_lengths = np.zeros(KFolds.n_splits, dtype=int)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        fold_lengths[fold_index] = len(test_indices)
        estimator.fit(X_train[train_indices], y_train[train_indices])
        test_folds_preds.append(estimator.predict(X_train[train_indices]))
    min_fold_length = fold_lengths.min()
    test_folds_preds = np.array(
        [test_fold_preds[:min_fold_length] for test_fold_preds in
         test_folds_preds])
    return test_folds_preds


class BaseMonoviewClassifier(BaseClassifier):

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multi_class=False):
        """
        Base method that returns an empty string if there is not interpretation
        method in the classifier's module
        """
        if hasattr(self, "feature_importances_"):
            return self.get_feature_importance(directory, base_file_name,
                                               feature_ids,)
        else:
            return ""


    def get_feature_importance(self, directory, base_file_name, feature_ids,
                               nb_considered_feats=50):
        """Used to generate a graph and a pickle dictionary representing
        feature importances"""
        feature_importances = self.feature_importances_
        sorted_args = np.argsort(-feature_importances)
        feature_importances_sorted = feature_importances[sorted_args][
            :nb_considered_feats]
        feature_indices_sorted = sorted_args[:nb_considered_feats]
        features_importances_dict = dict((feature_ids[feature_index], feature_importance)
                                         for feature_index, feature_importance in
                                         enumerate(feature_importances)
                                         if feature_importance != 0)
        with open(directory + 'feature_importances.pickle', 'wb') as handle:
            pickle.dump(features_importances_dict, handle)
        interpret_string = "Feature importances : \n"
        for feature_index, feature_importance in zip(feature_indices_sorted,
                                                   feature_importances_sorted):
            if feature_importance > 0:
                interpret_string += "- Feature : " + feature_ids[feature_index] + \
                                    ", feature importance : " + str(
                    feature_importance) + "\n"
        return interpret_string

    def get_name_for_fusion(self):
        return self.__class__.__name__[:4]


def percent(x, pos):
    """Used to print percentage of importance on the y axis"""
    return '%1.1f %%' % (x * 100)


class MonoviewResult(object):
    def __init__(self, view_index, classifier_name, view_name, metrics_scores,
                 full_labels_pred, classifier_config,
                 classifier, n_features, hps_duration, fit_duration,
                 pred_duration, class_metric_scores):
        self.view_index = view_index
        self.classifier_name = classifier_name
        self.view_name = view_name
        self.metrics_scores = metrics_scores
        self.full_labels_pred = full_labels_pred
        self.classifier_config = classifier_config
        self.clf = classifier
        self.n_features = n_features
        self.hps_duration = hps_duration
        self.fit_duration = fit_duration
        self.pred_duration = pred_duration
        self.class_metric_scores = class_metric_scores

    def get_classifier_name(self):
        return self.classifier_name + "-" + self.view_name


def get_accuracy_graph(plotted_data, classifier_name, file_name,
                       name="Accuracies", bounds=None, bound_name=None,
                       boosting_bound=None, set="train",
                       zero_to_one=True):  # pragma: no cover
    if not isinstance(name, str):
        name = " ".join(name.getConfig().strip().split(" ")[:2])
    f, ax = plt.subplots(nrows=1, ncols=1)
    if zero_to_one:
        ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_title(name + " during " + set + " for " + classifier_name)
    x = np.arange(len(plotted_data))
    scat = ax.scatter(x, np.array(plotted_data), marker=".")
    if bounds:
        if boosting_bound:
            scat2 = ax.scatter(x, boosting_bound, marker=".")
            scat3 = ax.scatter(x, np.array(bounds), marker=".", )
            ax.legend((scat, scat2, scat3),
                      (name, "Boosting bound", bound_name))
        else:
            scat2 = ax.scatter(x, np.array(bounds), marker=".", )
            ax.legend((scat, scat2),
                      (name, bound_name))
        # plt.tight_layout()
    else:
        ax.legend((scat,), (name,))
    f.savefig(file_name, transparent=True)
    plt.close()


class MonoviewResultAnalyzer(ResultAnalyser):

    def __init__(self, view_name, classifier_name, shape, classifier,
                 classification_indices, k_folds, hps_method, metrics_dict,
                 n_iter, class_label_names, pred,
                 directory, base_file_name, labels, database_name, nb_cores,
                 duration, feature_ids):
        ResultAnalyser.__init__(self, classifier, classification_indices,
                                k_folds, hps_method, metrics_dict, n_iter,
                                class_label_names, pred,
                                directory, base_file_name, labels,
                                database_name, nb_cores, duration, feature_ids)
        self.view_name = view_name
        self.classifier_name = classifier_name
        self.shape = shape

    def get_base_string(self):
        return "Classification on {} for {} with {}.\n\n".format(
            self.database_name, self.view_name, self.classifier_name
        )

    def get_view_specific_info(self):
        return "\t- View name : {}\t View shape : {}\n".format(self.view_name,
                                                               self.shape)
