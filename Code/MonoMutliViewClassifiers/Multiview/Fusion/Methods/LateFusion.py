#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
import itertools
from joblib import Parallel, delayed
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.svm import SVC
import os

import MonoviewClassifiers
from utils.Dataset import getV


def fitMonoviewClassifier(classifierName, data, labels, classifierConfig, needProbas):
    monoviewClassifier = getattr(MonoviewClassifiers, classifierName)
    if needProbas and not monoviewClassifier.canProbas():
        monoviewClassifier = getattr(MonoviewClassifiers, "DecisionTree")
    classifier = monoviewClassifier.fit(data,labels,**dict((str(configIndex), config) for configIndex, config in
                                      enumerate(classifierConfig
                                                )))
    return classifier


def getAccuracies(LateFusionClassifiers):
    return ""


def intersect(allClassifersNames, directory):
    wrongSets = []
    nbViews = 0
    for classifierIndex, classifierName in enumerate(allClassifersNames):
        wrongSets[classifierIndex]=[]
        classifierDirectory = directory+"/"+classifierName+"/"
        for viewIndex, viewDirectory in enumerate(os.listdir(classifierDirectory)):
            nbViews+=1
            for resultFileName in os.listdir(classifierDirectory+"/"+viewDirectory+"/"):
                if resultFileName.endswith("train_labels.csv"):
                    yTrainFileName = classifierDirectory+"/"+viewDirectory+"/"+resultFileName
                elif resultFileName.endswith("train_pred.csv"):
                    yTrainPredFileName = classifierDirectory+"/"+viewDirectory+"/"+resultFileName
            train = np.genfromtxt(yTrainFileName, delimiter=",").astype(np.int16)
            pred = np.genfromtxt(yTrainPredFileName, delimiter=",").astype(np.int16)
            length = len(train)
            wrongLabelsIndices = np.where(train+pred == 1)
            wrongSets[classifierIndex][viewIndex]=wrongLabelsIndices
    combinations = itertools.combinations_with_replacement(range(nbViews), len(allClassifersNames))
    bestLen = length
    bestCombination = None
    for combination in combinations:
        intersect = np.arange(length, dtype=np.int16)
        for viewIndex, classifierindex in enumerate(combination):
            intersect = np.intersect1d(intersect, wrongSets[classifierIndex][viewIndex])
        if len(intersect) < bestLen:
            bestLen = len(intersect)
            bestCombination = combination
    return [allClassifersNames[index] for index in bestCombination]




def getClassifiers(selectionMethodName, allClassifiersNames, directory):
    selectionMethod = locals()[selectionMethodName]
    classifiersNames = selectionMethod(allClassifiersNames, directory)
    return classifiersNames


class LateFusionClassifier(object):
    def __init__(self, randomState, monoviewClassifiersNames, monoviewClassifiersConfigs, monoviewSelection, NB_CORES=1):
        self.monoviewClassifiersNames = monoviewClassifiersNames
        self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
        self.monoviewClassifiers = []
        self.nbCores = NB_CORES
        self.accuracies = np.zeros(len(monoviewClassifiersNames))
        self.needProbas = False
        self.monoviewSelection = monoviewSelection
        self.randomState = randomState

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices == None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        monoviewSelectionMethod = locals()[self.monoviewSelection]
        self.monoviewClassifiers = monoviewSelectionMethod()
        self.monoviewClassifiers = Parallel(n_jobs=self.nbCores)(
            delayed(fitMonoviewClassifier)(self.monoviewClassifiersNames[index],
                                              getV(DATASET, viewIndex, trainIndices),
                                              DATASET.get("Labels")[trainIndices],
                                              self.monoviewClassifiersConfigs[index], self.needProbas)
            for index, viewIndex in enumerate(viewsIndices))