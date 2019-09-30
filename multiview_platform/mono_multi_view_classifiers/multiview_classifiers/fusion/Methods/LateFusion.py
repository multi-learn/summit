#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
import itertools
from joblib import Parallel, delayed
import sys
import math

from .... import monoview_classifiers
from .... import metrics
from ....utils.dataset import get_v


# def canProbasClassifier(classifierConfig):
#     try:
#         _ = getattr(classifierConfig, "predict_proba")
#         return True
#     except AttributeError:
#         return False


def fitMonoviewClassifier(monoviewClassifier, data, labels, needProbas, randomState, nbCores=1):
        if needProbas and not monoviewClassifier.canProbas():
            DTConfig = {"max_depth": 300, "criterion": "entropy", "splitter": "random"}
            monoviewClassifier = getattr(monoview_classifiers, "DecisionTree")(random_state=randomState, **DTConfig)
            classifier = monoviewClassifier.fit(data, labels)
            return classifier
        else:
            # if type(classifierConfig) is dict:
            #     pass
            # else:
            #     classifierConfig = dict((str(configIndex), config)
            #                              for configIndex, config in enumerate(classifierConfig))

            classifier = monoviewClassifier.fit(data, labels)
            return classifier


def getScores(LateFusionClassifiers):
    return ""


def intersect(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
    wrongSets = [[] for _ in viewsIndices]
    # wrongSets = [0 for _ in allClassifersNames]
    classifiersNames = [[] for _ in viewsIndices]
    nbViews = len(viewsIndices)
    trainLabels = np.genfromtxt(directory + "train_labels.csv", delimiter=",").astype(np.int16)
    length = len(trainLabels)
    for resultMonoview in resultsMonoview:
        if resultMonoview.classifier_name in classifiersNames[viewsIndices.index(resultMonoview.view_index)]:
            classifierIndex = classifiersNames.index(resultMonoview.classifier_name)
            wrongSets[resultMonoview.view_index][classifierIndex] = np.where(
                trainLabels + resultMonoview.full_labels_pred[classificationIndices[0]] == 1)[0]
        else:
            classifiersNames[viewsIndices.index(resultMonoview.view_index)].append(resultMonoview.classifier_name)
            wrongSets[viewsIndices.index(resultMonoview.view_index)].append(
                np.where(trainLabels + resultMonoview.full_labels_pred[classificationIndices[0]] == 1)[0])

    combinations = itertools.combinations_with_replacement(range(len(classifiersNames[0])), nbViews)
    bestLen = length
    bestCombination = None
    for combination in combinations:
        intersect = np.arange(length, dtype=np.int16)
        for viewIndex, classifierIndex in enumerate(combination):
            intersect = np.intersect1d(intersect, wrongSets[viewIndex][classifierIndex])
        if len(intersect) < bestLen:
            bestLen = len(intersect)
            bestCombination = combination
    return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(bestCombination)]


def bestScore(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    scores = np.zeros((nbViews, nbClassifiers))
    classifiersNames = [[] for _ in viewsIndices]
    metricName = resultsMonoview[0].metrics_scores.keys()[0]
    metricModule = getattr(metrics, metricName)
    if metricModule.getConfig()[-14] == "h":
        betterHigh = True
    else:
        betterHigh = False
    for resultMonoview in resultsMonoview:
        if resultMonoview.classifier_name not in classifiersNames[resultMonoview.view_index]:
            classifiersNames[resultMonoview.view_index].append(resultMonoview.classifier_name)
        classifierIndex = classifiersNames[resultMonoview.view_index].index(resultMonoview.classifier_name)
        scores[resultMonoview.view_index, classifierIndex] = resultMonoview.metrics_scores.values()[0][0]

    if betterHigh:
        classifierIndices = np.argmax(scores, axis=1)
    else:
        classifierIndices = np.argmin(scores, axis=1)
    return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(classifierIndices)]


def getClassifiers(selectionMethodName, allClassifiersNames, directory, viewsIndices, resultsMonoview,
                   classificationIndices):
    thismodule = sys.modules[__name__]
    selectionMethod = getattr(thismodule, selectionMethodName)
    classifiersNames = selectionMethod(allClassifiersNames, directory, viewsIndices, resultsMonoview,
                                       classificationIndices)
    return classifiersNames


def getConfig(classifiersNames, resultsMonoview, viewsIndices):
    classifiersConfigs = [0 for _ in range(len(classifiersNames))]
    for classifierIndex, classifierName in enumerate(classifiersNames):
        for resultMonoview in resultsMonoview:
            if resultMonoview.view_index == viewsIndices[classifierIndex] and resultMonoview.classifier_name == classifierName:
                classifiersConfigs[classifierIndex] = resultMonoview.classifier_config
    return classifiersConfigs


class LateFusionClassifier(object):
    def __init__(self, randomState, monoviewClassifiersNames, monoviewClassifiersConfigs, monoviewSelection,
                 NB_CORES=1):
        self.monoviewClassifiersNames = monoviewClassifiersNames
        monoviewClassifiersModules = [getattr(monoview_classifiers, classifierName)
                                      for classifierName in self.monoviewClassifiersNames]
        if type(monoviewClassifiersConfigs[0]) == dict:
            self.monoviewClassifiers = [
                getattr(monoviewClassifiersModule, classifierName)(random_state=randomState, **config)
                for monoviewClassifiersModule, config, classifierName
                in zip(monoviewClassifiersModules, monoviewClassifiersConfigs, monoviewClassifiersNames)]
        else:
            self.monoviewClassifiers = [
                getattr(monoviewClassifiersModule, classifierName)(random_state=randomState,)
                for monoviewClassifiersModule, config, classifierName
                in zip(monoviewClassifiersModules, monoviewClassifiersConfigs, monoviewClassifiersNames)]
        self.nbCores = NB_CORES
        self.accuracies = np.zeros(len(monoviewClassifiersNames))
        self.needProbas = False
        self.monoviewSelection = monoviewSelection
        self.randomState = randomState

    def fit_hdf5(self, DATASET, labels, trainIndices=None, viewsIndices=None):
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices is None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        self.monoviewClassifiers = Parallel(n_jobs=self.nbCores)(
                delayed(fitMonoviewClassifier)(self.monoviewClassifiers[index],
                                               get_v(DATASET, viewIndex, trainIndices),
                                               labels[trainIndices],
                                               self.needProbas, self.randomState)
                for index, viewIndex in enumerate(viewsIndices))
