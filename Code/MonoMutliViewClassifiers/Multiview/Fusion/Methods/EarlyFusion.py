#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np


class EarlyFusionClassifier(object):
    def __init__(self, monoviewClassifierName, monoviewClassifierConfig, NB_CORES=1):
        self.monoviewClassifierName = monoviewClassifierName[0]
        self.monoviewClassifiersConfig = monoviewClassifierConfig[0]
        self.monoviewClassifier = None
        self.nbCores = NB_CORES
        self.monoviewData = None

    def makeMonoviewData_hdf5(self, DATASET, weights=None, usedIndices=None):
        if not usedIndices:
            uesdIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
        if weights== None:
            weights = np.array([1/NB_VIEW for i in range(NB_VIEW)])
        if sum(weights)!=1:
            weights = weights/sum(weights)
        self.monoviewData = np.concatenate([weights[viewIndex]*DATASET.get("View"+str(viewIndex))[usedIndices, :]
                                                         for viewIndex in np.arange(NB_VIEW)], axis=1)

