#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
import itertools
from joblib import Parallel, delayed
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.svm import SVC
import os
import sys

import MonoviewClassifiers
import Metrics
from utils.Dataset import getV


def canProbasClassifier(classifierConfig):
    try:
        _ = getattr(classifierConfig, "predict_proba")
        return True
    except AttributeError:
        return False


def fitMonoviewClassifier(classifierName, data, labels, classifierConfig, needProbas, randomState):
    if type(classifierConfig) == dict:
        monoviewClassifier = getattr(MonoviewClassifiers, classifierName)
        if needProbas and not monoviewClassifier.canProbas():
            monoviewClassifier = getattr(MonoviewClassifiers, "DecisionTree")
            DTConfig = {"0":300, "1":"entropy", "2":"random"}
            classifier = monoviewClassifier.fit(data,labels, randomState,DTConfig)
            return classifier
        else:
            classifier = monoviewClassifier.fit(data,labels, randomState,**dict((str(configIndex), config) for configIndex, config in
                                              enumerate(classifierConfig
                                                        )))
            return classifier


def getScores(LateFusionClassifiers):
    return ""


def intersect(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
    wrongSets = [[] for _ in viewsIndices]
    # wrongSets = [0 for _ in allClassifersNames]
    classifiersNames = [[] for _ in viewsIndices]
    nbViews = len(viewsIndices)
    trainLabels = np.genfromtxt(directory+"train_labels.csv", delimiter=",").astype(np.int16)
    length = len(trainLabels)
    for resultMonoview in resultsMonoview:
        if resultMonoview[1][0] in classifiersNames[resultMonoview[0]]:
            classifierIndex = classifiersNames.index(resultMonoview[1][0])
            wrongSets[resultMonoview[0]][classifierIndex] = np.where(trainLabels+resultMonoview[1][3][classificationIndices[0]] == 1)
        else:
            classifiersNames[resultMonoview[0]].append(resultMonoview[1][0])
            wrongSets[resultMonoview[0]].append(np.where(trainLabels+resultMonoview[1][3][classificationIndices[0]] == 1))
    # for classifierIndex, classifierName in enumerate(allClassifersNames):
    #     try:
    #         classifierDirectory = directory+classifierName+"/"
    #         viewDirectoryNames = os.listdir(classifierDirectory)
    #         wrongSets[classifierIndex]=[0 for _ in viewDirectoryNames]
    #         for viewIndex, viewDirectoryName in enumerate(viewDirectoryNames):
    #             for resultFileName in os.listdir(classifierDirectory+"/"+viewDirectoryName+"/"):
    #                 if resultFileName.endswith("train_labels.csv"):
    #                     yTrainFileName = classifierDirectory+"/"+viewDirectoryName+"/"+resultFileName
    #                 elif resultFileName.endswith("train_pred.csv"):
    #                     yTrainPredFileName = classifierDirectory+"/"+viewDirectoryName+"/"+resultFileName
    #             train = np.genfromtxt(yTrainFileName, delimiter=",").astype(np.int16)
    #             pred = np.genfromtxt(yTrainPredFileName, delimiter=",").astype(np.int16)
    #             length = len(train)
    #             wrongLabelsIndices = np.where(train+pred == 1)
    #             wrongSets[classifierIndex][viewIndex]=wrongLabelsIndices
    #     except OSError:
    #         for viewIndex in range(nbViews):
    #             wrongSets[classifierIndex][viewIndex]= np.arange(length)

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


# def getFormFile(directory, viewDirectory, resultFileName):
#     file = open(directory+"/"+viewDirectory+"/"+resultFileName)
#     for line in file:
#         if "Score on train" in line:
#             score = float(line.strip().split(":")[1])
#             break
#         elif "train" in line:
#             metricName = line.strip().split(" ")[0]
#     metricModule = getattr(Metrics, metricName)
#     if metricModule.getConfig()[-14]=="h":
#         betterHigh = True
#     else:
#         betterHigh = False
#     return score, betterHigh


def bestScore(allClassifersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
    nbViews = len(viewsIndices)
    nbClassifiers = len(allClassifersNames)
    scores = np.zeros((nbViews, nbClassifiers))
    classifiersNames = [[] for _ in viewsIndices]
    metricName = resultsMonoview[0][1][2].keys()[0]
    metricModule = getattr(Metrics, metricName)
    if metricModule.getConfig()[-14]=="h":
        betterHigh = True
    else:
        betterHigh = False
    for resultMonoview in resultsMonoview:
        if resultMonoview[1][0] not in classifiersNames[resultMonoview[0]]:
            classifiersNames[resultMonoview[0]].append(resultMonoview[1][0])
        classifierIndex = classifiersNames[resultMonoview[0]].index(resultMonoview[1][0])
        scores[resultMonoview[0],classifierIndex] = resultMonoview[1][2].values()[0][0]
    #
    # for classifierIndex, classifierName in enumerate(allClassifersNames):
    #     classifierDirectory = directory+"/"+classifierName+"/"
    #     for viewIndex, viewDirectory in enumerate(os.listdir(classifierDirectory)):
    #         for resultFileName in os.listdir(classifierDirectory+"/"+viewDirectory+"/"):
    #             if resultFileName.endswith(".txt"):
    #                 scores[viewIndex, classifierIndex], betterHigh = getFormFile(directory, viewDirectory, resultFileName)
    if betterHigh:
        classifierIndices = np.argmax(scores, axis=1)
    else:
        classifierIndices = np.argmin(scores, axis=1)
    return [classifiersNames[viewIndex][index] for viewIndex, index in enumerate(classifierIndices)]


def getClassifiers(selectionMethodName, allClassifiersNames, directory, viewsIndices, resultsMonoview, classificationIndices):
    thismodule = sys.modules[__name__]
    selectionMethod = getattr(thismodule, selectionMethodName)
    classifiersNames = selectionMethod(allClassifiersNames, directory, viewsIndices, resultsMonoview, classificationIndices)
    return classifiersNames


def getConfig(classifiersNames, resultsMonoview):
    classifiersConfigs = [0 for _ in range(len(classifiersNames))]
    for viewIndex, classifierName in enumerate(classifiersNames):
        for resultMonoview in resultsMonoview:
            if resultMonoview[0]==viewIndex and resultMonoview[1][0]==classifierName:
                classifiersConfigs[viewIndex]=resultMonoview[1][4]
    return classifiersConfigs


class LateFusionClassifier(object):
    def __init__(self, randomState, monoviewClassifiersNames, monoviewClassifiersConfigs, monoviewSelection, NB_CORES=1):
        self.monoviewClassifiersNames = monoviewClassifiersNames
        if type(monoviewClassifiersConfigs[0])==dict:
            self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
            self.monoviewClassifiers = []
        else:
            self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
        self.nbCores = NB_CORES
        self.accuracies = np.zeros(len(monoviewClassifiersNames))
        self.needProbas = False
        self.monoviewSelection = monoviewSelection
        self.randomState = randomState

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices is None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])

        self.monoviewClassifiers = Parallel(n_jobs=self.nbCores)(
                delayed(fitMonoviewClassifier)(self.monoviewClassifiersNames[index],
                                                  getV(DATASET, viewIndex, trainIndices),
                                                  DATASET.get("Labels").value[trainIndices],
                                                  self.monoviewClassifiersConfigs[index], self.needProbas, self.randomState)
                for index, viewIndex in enumerate(viewsIndices))