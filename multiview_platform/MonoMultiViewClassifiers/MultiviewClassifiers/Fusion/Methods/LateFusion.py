#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
import itertools
from joblib import Parallel, delayed
import sys
import math

from .... import MonoviewClassifiers
from .... import Metrics
from ....utils.Dataset import getV


def canProbasClassifier(classifierConfig):
    try:
        _ = getattr(classifierConfig, "predict_proba")
        return True
    except AttributeError:
        return False


def fitMonoviewClassifier(classifierName, data, labels, classifierConfig, needProbas, randomState, nbCores=1):
    if type(classifierConfig) == dict:
        monoviewClassifier = getattr(MonoviewClassifiers, classifierName)
        if needProbas and not monoviewClassifier.canProbas():
            monoviewClassifier = getattr(MonoviewClassifiers, "DecisionTree")
            DTConfig = {"max_depth": 300, "criterion": "entropy", "splitter": "random"}
            classifier = monoviewClassifier.fit(data, labels, randomState, nbCores, **DTConfig)
            return classifier
        else:
            if type(classifierConfig) is dict:
                pass
            else:
                classifierConfig = dict((str(configIndex), config)
                                         for configIndex, config in enumerate(classifierConfig))

            classifier = monoviewClassifier.fit(data, labels, randomState, nbCores, **classifierConfig)
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
        if resultMonoview[1][0] in classifiersNames[resultMonoview[0]]:
            classifierIndex = classifiersNames.index(resultMonoview[1][0])
            wrongSets[resultMonoview[0]][classifierIndex] = np.where(
                trainLabels + resultMonoview[1][3][classificationIndices[0]] == 1)[0]
        else:
            classifiersNames[resultMonoview[0]].append(resultMonoview[1][0])
            wrongSets[resultMonoview[0]].append(
                np.where(trainLabels + resultMonoview[1][3][classificationIndices[0]] == 1)[0])

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


# def getClassifiersDecisions(allClassifersNames, viewsIndices, resultsMonoview):
#     nbViews = len(viewsIndices)
#     nbClassifiers = len(allClassifersNames)
#     nbFolds = len(resultsMonoview[0][1][6])
#     foldsLen = len(resultsMonoview[0][1][6][0])
#     classifiersNames = [[] for _ in viewsIndices]
#     classifiersDecisions = np.zeros((nbViews, nbClassifiers, nbFolds, foldsLen))
#
#     for resultMonoview in resultsMonoview:
#         if resultMonoview[1][0] in classifiersNames[viewsIndices.index(resultMonoview[0])]:
#             pass
#         else:
#             classifiersNames[viewsIndices.index(resultMonoview[0])].append(resultMonoview[1][0])
#         classifierIndex = classifiersNames[viewsIndices.index(resultMonoview[0])].index(resultMonoview[1][0])
#         classifiersDecisions[viewsIndices.index(resultMonoview[0]), classifierIndex] = resultMonoview[1][6]
#     return classifiersDecisions, classifiersNames
#
#
# def disagreement(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
#
#     classifiersDecisions, classifiersNames = getClassifiersDecisions(allClassifersNames, viewsIndices, resultsMonoview)
#
#     foldsLen = len(resultsMonoview[0][1][6][0])
#     nbViews = len(viewsIndices)
#     nbClassifiers = len(allClassifersNames)
#     combinations = itertools.combinations_with_replacement(range(nbClassifiers), nbViews)
#     nbCombinations = math.factorial(nbClassifiers+nbViews-1) / math.factorial(nbViews) / math.factorial(nbClassifiers-1)
#     disagreements = np.zeros(nbCombinations)
#     combis = np.zeros((nbCombinations, nbViews), dtype=int)
#
#     for combinationsIndex, combination in enumerate(combinations):
#         combis[combinationsIndex] = combination
#         combiWithView = [(viewIndex,combiIndex) for viewIndex, combiIndex in enumerate(combination)]
#         binomes = itertools.combinations(combiWithView, 2)
#         nbBinomes = math.factorial(nbViews) / 2 / math.factorial(nbViews-2)
#         disagreement = np.zeros(nbBinomes)
#         for binomeIndex, binome in enumerate(binomes):
#             (viewIndex1, classifierIndex1), (viewIndex2, classifierIndex2) = binome
#             nbDisagree = np.sum(np.logical_xor(classifiersDecisions[viewIndex1, classifierIndex1],
#                                                classifiersDecisions[viewIndex2, classifierIndex2])
#                                 , axis=1)/foldsLen
#             disagreement[binomeIndex] = np.mean(nbDisagree)
#         disagreements[combinationsIndex] = np.mean(disagreement)
#     print(disagreements)
#     bestCombiIndex = np.argmax(disagreements)
#     bestCombination = combis[bestCombiIndex]
#
#     return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(bestCombination)]








# def allMonoviewClassifiers(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
#     return allClassifersNames


def bestScore(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    scores = np.zeros((nbViews, nbClassifiers))
    classifiersNames = [[] for _ in viewsIndices]
    metricName = resultsMonoview[0][1][2].keys()[0]
    metricModule = getattr(Metrics, metricName)
    if metricModule.getConfig()[-14] == "h":
        betterHigh = True
    else:
        betterHigh = False
    for resultMonoview in resultsMonoview:
        if resultMonoview[1][0] not in classifiersNames[resultMonoview[0]]:
            classifiersNames[resultMonoview[0]].append(resultMonoview[1][0])
        classifierIndex = classifiersNames[resultMonoview[0]].index(resultMonoview[1][0])
        scores[resultMonoview[0], classifierIndex] = resultMonoview[1][2].values()[0][0]

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


def getConfig(classifiersNames, resultsMonoview):
    classifiersConfigs = [0 for _ in range(len(classifiersNames))]
    for viewIndex, classifierName in enumerate(classifiersNames):
        for resultMonoview in resultsMonoview:
            if resultMonoview[0] == viewIndex and resultMonoview[1][0] == classifierName:
                classifiersConfigs[viewIndex] = resultMonoview[1][4]
    return classifiersConfigs


class LateFusionClassifier(object):
    def __init__(self, randomState, monoviewClassifiersNames, monoviewClassifiersConfigs, monoviewSelection,
                 NB_CORES=1):
        self.monoviewClassifiersNames = monoviewClassifiersNames
        if type(monoviewClassifiersConfigs[0]) == dict:
            self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
            self.monoviewClassifiers = []
        else:
            self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
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
                delayed(fitMonoviewClassifier)(self.monoviewClassifiersNames[index],
                                               getV(DATASET, viewIndex, trainIndices),
                                               labels[trainIndices],
                                               self.monoviewClassifiersConfigs[index], self.needProbas, self.randomState)
                for index, viewIndex in enumerate(viewsIndices))
