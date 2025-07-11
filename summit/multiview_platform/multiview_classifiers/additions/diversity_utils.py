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
import itertools
import math

import numpy as np

from .fusion_utils import BaseFusionClassifier
from ...multiview.multiview_utils import ConfigGenerator, \
    get_available_monoview_classifiers, \
    BaseMultiviewClassifier
from ...utils.dataset import get_samples_views_indices


class DiversityFusionClassifier(BaseMultiviewClassifier,
                                BaseFusionClassifier):
    """This is the base class for all the diversity fusion based classifiers."""

    def __init__(self, random_state=None, classifier_names=None,
                 monoview_estimators=None, classifier_configs=None):
        """Used to init the instances"""
        BaseMultiviewClassifier.__init__(self, random_state)
        if classifier_names is None:
            classifier_names = get_available_monoview_classifiers()
        self.classifier_names = classifier_names
        self.param_names = ["classifier_configs"]
        self.distribs = [ConfigGenerator(get_available_monoview_classifiers())]
        self.monoview_estimators = monoview_estimators
        self.classifier_configs = classifier_configs

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_samples_views_indices(X,
                                                                train_indices,
                                                                view_indices)
        self.used_views = view_indices
        # TODO : Finer analysis, may support a bit of mutliclass
        if np.unique(y[train_indices]).shape[0] > 2:
            raise ValueError(
                "Multiclass not supported, classes used : {}".format(
                    np.unique(y[train_indices])))
        if self.monoview_estimators is None:
            self.monoview_estimators = []
            for classifier_idx, classifier_name in enumerate(
                    self.classifier_names):
                self.monoview_estimators.append([])
                for idx, view_idx in enumerate(view_indices):
                    estimator = self.init_monoview_estimator(classifier_name,
                                                             self.classifier_configs)
                    estimator.fit(X.get_v(view_idx, train_indices),
                                  y[train_indices])
                    self.monoview_estimators[classifier_idx].append(estimator)
        else:
            pass  # TODO
        self.choose_combination(X, y, train_indices, view_indices)
        return self

    def predict(self, X, sample_indices=None, view_indices=None):
        """Just a weighted majority vote"""
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        nb_class = X.get_nb_class()
        if nb_class > 2:
            nb_class = 3
        votes = np.zeros((len(sample_indices), nb_class), dtype=float)
        monoview_predictions = [
            monoview_estimator.predict(X.get_v(view_idx, sample_indices))
            for view_idx, monoview_estimator
            in zip(view_indices, self.monoview_estimators)]
        for idx, sample_index in enumerate(sample_indices):
            for monoview_estimator_index, monoview_prediciton in enumerate(
                    monoview_predictions):
                if int(monoview_prediciton[idx]) == -100:
                    votes[idx, 2] += 1
                else:
                    votes[idx, int(monoview_prediciton[idx])] += 1
        predicted_labels = np.argmax(votes, axis=1)
        return predicted_labels

    def get_classifiers_decisions(self, X, view_indices, samples_indices):
        classifiers_decisions = np.zeros((len(self.monoview_estimators),
                                          len(view_indices),
                                          len(samples_indices)))
        for estimator_idx, estimator in enumerate(self.monoview_estimators):
            for idx, view_index in enumerate(view_indices):
                classifiers_decisions[estimator_idx, idx, :] = estimator[
                    idx].predict(X.get_v(view_index, samples_indices))
        return classifiers_decisions

    def init_combinations(self, X, sample_indices, view_indices):
        classifiers_decisions = self.get_classifiers_decisions(X, view_indices,
                                                               sample_indices)
        nb_classifiers, nb_views, n_samples = classifiers_decisions.shape
        combinations = itertools.combinations_with_replacement(
            range(nb_classifiers),
            nb_views)
        nb_combinations = int(
            math.factorial(nb_classifiers + nb_views - 1) / math.factorial(
                nb_views) / math.factorial(
                nb_classifiers - 1))
        div_measure = np.zeros(nb_combinations)
        combis = np.zeros((nb_combinations, nb_views), dtype=int)
        return combinations, combis, div_measure, classifiers_decisions, nb_views


class GlobalDiversityFusionClassifier(DiversityFusionClassifier):

    def choose_combination(self, X, y, samples_indices, view_indices):
        combinations, combis, div_measure, classifiers_decisions, nb_views = self.init_combinations(
            X, samples_indices, view_indices)
        for combinationsIndex, combination in enumerate(combinations):
            combis[combinationsIndex] = combination
            div_measure[combinationsIndex] = self.diversity_measure(
                classifiers_decisions,
                combination,
                y[samples_indices])
        best_combi_index = np.argmax(div_measure)
        best_combination = combis[best_combi_index]
        self.monoview_estimators = [
            self.monoview_estimators[classifier_index][view_index]
            for view_index, classifier_index
            in enumerate(best_combination)]


class CoupleDiversityFusionClassifier(DiversityFusionClassifier):

    def choose_combination(self, X, y, samples_indices, view_indices):
        combinations, combis, div_measure, classifiers_decisions, nb_views = self.init_combinations(
            X, samples_indices, view_indices)
        for combinations_index, combination in enumerate(combinations):
            combis[combinations_index] = combination
            combi_with_view = [(viewIndex, combiIndex) for viewIndex, combiIndex
                               in
                               enumerate(combination)]
            binomes = itertools.combinations(combi_with_view, 2)
            nb_binomes = int(
                math.factorial(nb_views) / 2 / math.factorial(nb_views - 2))
            couple_diversities = np.zeros(nb_binomes)
            for binome_index, binome in enumerate(binomes):
                (view_index_1, classifier_index_1), (
                    view_index_2, classifier_index_2) = binome
                couple_diversity = np.mean(
                    self.diversity_measure(
                        classifiers_decisions[classifier_index_1,
                                              view_index_1],
                        classifiers_decisions[classifier_index_2,
                                              view_index_2],
                        y[samples_indices])
                )
                couple_diversities[binome_index] = couple_diversity
            div_measure[combinations_index] = np.mean(couple_diversities)
        best_combi_index = np.argmax(div_measure)
        best_combination = combis[best_combi_index]
        self.monoview_estimators = [
            self.monoview_estimators[classifier_index][view_index]
            for view_index, classifier_index
            in enumerate(best_combination)]

#
# def CQ_div_measure(classifiersNames, classifiersDecisions, measurement,
#                    foldsGroudTruth):
#     """
#     This function is used to measure a pseudo-CQ measurement based on the minCq algorithm.
#     It's a mix between couple_div_measure and global_div_measure that uses multiple measurements.
#     """
#     nbViews, nbClassifiers, nbFolds, foldsLen = classifiersDecisions.shape
#     combinations = itertools.combinations_with_replacement(range(nbClassifiers),
#                                                            nbViews)
#     nbCombinations = int(
#         math.factorial(nbClassifiers + nbViews - 1) / math.factorial(
#             nbViews) / math.factorial(nbClassifiers - 1))
#     div_measure = np.zeros(nbCombinations)
#     combis = np.zeros((nbCombinations, nbViews), dtype=int)
#
#     for combinationsIndex, combination in enumerate(combinations):
#         combis[combinationsIndex] = combination
#         combiWithView = [(viewIndex, combiIndex) for viewIndex, combiIndex in
#                          enumerate(combination)]
#         binomes = itertools.combinations(combiWithView, 2)
#         nbBinomes = int(
#             math.factorial(nbViews) / 2 / math.factorial(nbViews - 2))
#         disagreement = np.zeros(nbBinomes)
#         div_measure[combinationsIndex] = measurement[1](classifiersDecisions,
#                                                         combination,
#                                                         foldsGroudTruth,
#                                                         foldsLen)
#         for binomeIndex, binome in enumerate(binomes):
#             (viewIndex1, classifierIndex1), (
#             viewIndex2, classifierIndex2) = binome
#             nbDisagree = np.sum(measurement[0](
#                 classifiersDecisions[viewIndex1, classifierIndex1],
#                 classifiersDecisions[viewIndex2, classifierIndex2],
#                 foldsGroudTruth)
#                                 , axis=1) / float(foldsLen)
#             disagreement[binomeIndex] = np.mean(nbDisagree)
#         div_measure[combinationsIndex] /= float(np.mean(disagreement))
#     bestCombiIndex = np.argmin(div_measure)
#     bestCombination = combis[bestCombiIndex]
#
#     return [classifiersNames[viewIndex][index] for viewIndex, index in
#             enumerate(bestCombination)], div_measure[
#                bestCombiIndex]
#
