#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
from utils.Dataset import getV


class EarlyFusionClassifier(object):
    def __init__(self, randomState, monoviewClassifierName, monoviewClassifierConfig, NB_CORES=1):
        self.monoviewClassifierName = monoviewClassifierName
        if type(monoviewClassifierConfig) == dict:
            pass
        elif monoviewClassifierConfig is None:
            pass
        else:
            monoviewClassifierConfig = dict((str(configIndex), config[0]) for configIndex, config in
                                            enumerate(monoviewClassifierConfig
                                                      ))
        self.monoviewClassifiersConfig = monoviewClassifierConfig
        self.monoviewClassifier = None
        self.nbCores = NB_CORES
        self.monoviewData = None
        self.randomState = randomState

    def makeMonoviewData_hdf5(self, DATASET, weights=None, usedIndices=None, viewsIndices=None):
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if type(weights) == type(None):
            weights = np.array([1 / nbView for i in range(nbView)])
        if sum(weights) != 1:
            weights = weights / sum(weights)
        self.monoviewData = np.concatenate([getV(DATASET, viewIndex, usedIndices)
                                            for index, viewIndex in enumerate(viewsIndices)], axis=1)
