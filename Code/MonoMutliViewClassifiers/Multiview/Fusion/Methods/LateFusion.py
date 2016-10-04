#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
from joblib import Parallel, delayed
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

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


class LateFusionClassifier(object):
    def __init__(self, monoviewClassifiersNames, monoviewClassifiersConfigs, NB_CORES=1):
        self.monoviewClassifiersNames = monoviewClassifiersNames
        self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
        self.monoviewClassifiers = []
        self.nbCores = NB_CORES
        self.accuracies = np.zeros(len(monoviewClassifiersNames))
        self.needProbas = False

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices == None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        self.monoviewClassifiers = Parallel(n_jobs=self.nbCores)(
            delayed(fitMonoviewClassifier)(self.monoviewClassifiersNames[index],
                                              getV(DATASET, viewIndex, trainIndices),
                                              DATASET.get("Labels")[trainIndices],
                                              self.monoviewClassifiersConfigs[index], self.needProbas)
            for index, viewIndex in enumerate(viewsIndices))