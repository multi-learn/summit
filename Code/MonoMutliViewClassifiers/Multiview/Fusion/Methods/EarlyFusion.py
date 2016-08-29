#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np

import MonoviewClassifiers


class EarlyFusionClassifier(object):
    def __init__(self, monoviewClassifiersNames, monoviewClassifiersConfigs, NB_CORES=1):
        self.monoviewClassifierName = monoviewClassifiersNames[0]
        self.monoviewClassifiersConfig = monoviewClassifiersConfigs[0]
        self.monoviewClassifier = None
        self.nbCores = NB_CORES
        self.monoviewData = None

    def makeMonoviewData_hdf5(self, DATASET, weights=None, usedIndices=None):
        if not usedIndices:
            uesdIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
        if type(weights)=="NoneType":
            weights = np.array([1/NB_VIEW for i in range(NB_VIEW)])
        if sum(weights)!=1:
            weights = weights/sum(weights)
        self.monoviewData = np.concatenate([weights[viewIndex]*DATASET.get("View"+str(viewIndex))[usedIndices, :]
                                                         for viewIndex in np.arange(NB_VIEW)], axis=1)



# class WeightedLinear(EarlyFusionClassifier):
#     def __init__(self, NB_CORES=1, **kwargs):
#         EarlyFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['monoviewClassifiersConfigs'],
#                                       NB_CORES=NB_CORES)
#         self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))
#
#     def fit_hdf5(self, DATASET, trainIndices=None):
#         if not trainIndices:
#             trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
#         self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=trainIndices)
#         monoviewClassifierModule = getattr(MonoviewClassifiers, self.monoviewClassifierName)
#         desc, self.monoviewClassifier = monoviewClassifierModule.fit(self.monoviewData, DATASET.get("labels")[trainIndices],
#                                                                NB_CORES=self.nbCores,
#                                                                **dict((str(configIndex),config) for configIndex,config in
#                                                                       enumerate(self.monoviewClassifiersConfig)))
#
#     def predict_hdf5(self, DATASET, usedIndices=None):
#         if usedIndices == None:
#             usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
#         if usedIndices:
#             self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=usedIndices)
#             predictedLabels = self.monoviewClassifier.predict(self.monoviewData)
#         else:
#             predictedLabels=[]
#         return predictedLabels
#
#     def predict_proba_hdf5(self, DATASET, usedIndices=None):
#         if usedIndices == None:
#             usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
#         if usedIndices:
#             self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=usedIndices)
#             predictedLabels = self.monoviewClassifier.predict_proba(self.monoviewData)
#         else:
#             predictedLabels=[]
#         return predictedLabels
#
#     def getConfig(self, fusionMethodConfig ,monoviewClassifiersNames, monoviewClassifiersConfigs):
#         configString = "with weighted concatenation, using weights : "+", ".join(map(str, self.weights))+\
#                        " with monoview classifier : "
#         monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifiersNames[0])
#         configString += monoviewClassifierModule.getConfig(monoviewClassifiersConfigs[0])
#         return configString
#
#     def gridSearch(self, classificationKWARGS):
#
#         return

