import itertools
import math
import inspect
import os

import numpy as np

from ...multiview.multiview_utils import ConfigGenerator, \
    get_examples_views_indices, get_available_monoview_classifiers, \
    BaseMultiviewClassifier
from .fusion_utils import BaseFusionClassifier


class DiversityFusionClassifier(BaseMultiviewClassifier,
                                BaseFusionClassifier):
    """This is the base class for all the diversity fusion based classifiers."""

    def __init__(self, random_state=None, classifier_names=None,
                 monoview_estimators=None, classifier_configs=None):
        """Used to init the instances"""
        super(DiversityFusionClassifier, self).__init__(random_state)
        if classifier_names is None:
            classifier_names = get_available_monoview_classifiers()
        self.classifier_names = classifier_names
        self.param_names = ["classifier_configs"]
        self.distribs = [ConfigGenerator(get_available_monoview_classifiers())]
        self.estimator_pool = monoview_estimators
        self.classifier_configs = classifier_configs

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        if self.estimator_pool is None:
            self.estimator_pool = []
            for classifier_idx, classifier_name in enumerate(self.classifier_names):
                self.estimator_pool.append([])
                for idx, view_idx in enumerate(view_indices):
                    estimator = self.init_monoview_estimator(classifier_name, self.classifier_configs)
                    estimator.fit(X.get_v(view_idx, train_indices), y[train_indices])
                    self.estimator_pool[classifier_idx].append(estimator)
        else:
            pass #TODO
        self.choose_combination(X, y, train_indices, view_indices)
        return self

    def predict(self, X, example_indices=None, view_indices=None):
        """Just a weighted majority vote"""
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        nb_class = X.get_nb_class()
        if nb_class>2:
            nb_class=3
        votes = np.zeros((len(example_indices), nb_class), dtype=float)
        monoview_predictions = [monoview_estimator.predict(X.get_v(view_idx, example_indices))
                                for view_idx, monoview_estimator
                                in zip(view_indices, self.monoview_estimators)]
        for idx, example_index in enumerate(example_indices):
            for monoview_estimator_index, monoview_prediciton in enumerate(monoview_predictions):
                if int(monoview_prediciton[idx]) == -100:
                    votes[idx, 2] += 1
                else:
                    votes[idx, int(monoview_prediciton[idx])] += 1
        predicted_labels = np.argmax(votes, axis=1)
        return predicted_labels

    def get_classifiers_decisions(self, X, view_indices, examples_indices):
        classifiers_decisions = np.zeros((len(self.estimator_pool),
                                              len(view_indices),
                                              len(examples_indices)))
        for estimator_idx, estimator in enumerate(self.estimator_pool):
            for idx, view_index in enumerate(view_indices):
                classifiers_decisions[estimator_idx, idx, :] = estimator[
                    idx].predict(X.get_v(view_index, examples_indices))
        return classifiers_decisions

    def init_combinations(self, X, example_indices, view_indices):
        classifiers_decisions = self.get_classifiers_decisions(X, view_indices,
                                                               example_indices)
        nb_classifiers, nb_views, n_examples = classifiers_decisions.shape
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

    def choose_combination(self, X, y, examples_indices, view_indices):
        combinations, combis, div_measure, classifiers_decisions, nb_views = self.init_combinations(
            X, examples_indices, view_indices)
        for combinationsIndex, combination in enumerate(combinations):
            combis[combinationsIndex] = combination
            div_measure[combinationsIndex] = self.diversity_measure(
                classifiers_decisions,
                combination,
                y[examples_indices])
        best_combi_index = np.argmax(div_measure)
        best_combination = combis[best_combi_index]
        self.monoview_estimators = [self.estimator_pool[classifier_index][view_index]
                                    for view_index, classifier_index
                                    in enumerate(best_combination)]


class CoupleDiversityFusionClassifier(DiversityFusionClassifier):

    def choose_combination(self, X, y, examples_indices, view_indices):
        combinations, combis, div_measure, classifiers_decisions, nb_views = self.init_combinations(
            X, examples_indices, view_indices)
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
                        classifiers_decisions[classifier_index_1, view_index_1],
                        classifiers_decisions[classifier_index_2, view_index_2],
                        y[examples_indices])
                    )
                couple_diversities[binome_index] = couple_diversity
            div_measure[combinations_index] = np.mean(couple_diversities)
        best_combi_index = np.argmax(div_measure)
        best_combination = combis[best_combi_index]
        self.monoview_estimators = [self.estimator_pool[classifier_index][view_index]
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
