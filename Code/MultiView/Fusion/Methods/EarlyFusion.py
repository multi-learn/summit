#!/usr/bin/env python
# -*- encoding: utf-8

from MonoviewClassifiers import *

import numpy as np


class EarlyFusionClassifier:
    def __init__(self, monoviewClassifiersNames, monoviewClassifiersConfigs, NB_CORES=1):
        self.monoviewClassifiersName = monoviewClassifiersNames[0]
        self.monoviewClassifiersConfig = monoviewClassifiersConfigs[0]
        self.monoviewClassifier = None
        self.nbCores = NB_CORES
        self.monoviewData = None

    def makeMonoviewData(self, DATASET, weights=None):
        NB_VIEW = DATASET.get("nbView").value
        DATASET_LENGTH = DATASET.get("datasetLength").value
        if weights==None:
            weights = np.array([1/NB_VIEW for i in range(NB_VIEW)])
        if sum(weights)!=1:
            weights = weights/sum(weights)
        self.monoviewData = np.concatenate([weights[viewIndex]*DATASET["/View"+str(viewIndex)+"/matrix"][...]
                                                         for viewIndex in np.arange(NB_VIEW)], axis=1)


class WeightedLinear(EarlyFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        EarlyFusionClassifier.__init__(self, kwargs['monoviewClassifiersNames'], kwargs['monoviewClassifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))

    def fit_hdf5(self, DATASET, trainIndices=None):
        if not trainIndices:
            trainIndices = range(DATASET.get("datasetLength").value)
        self.makeMonoviewData(DATASET, weights=self.weights)
        monoviewClassifierModule = globals()[self.monoviewClassifiersName]
        self.monoviewClassifier = monoviewClassifierModule.fit(self.monoviewData, DATASET["/Labels/labelsArray"][...],
                                                               NB_CORES=self.nbCores,
                                                               **dict((str(configIndex),config) for configIndex,config in
                                                                      enumerate(self.monoviewClassifiersConfig)))

    def predict_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("datasetLength").value)
        if usedIndices:
            self.makeMonoviewData(DATASET, weights=self.weights)
            predictedLabels = self.monoviewClassifier.predict(self.monoviewData)
        else:
            predictedLabels=[]
        return predictedLabels

