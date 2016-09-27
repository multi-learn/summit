from Methods import *
import MonoviewClassifiers
import numpy as np
import logging
from utils.Dataset import getV


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype



def makeMonoviewData_hdf5(DATASET, weights=None, usedIndices=None, viewsIndices=None):
    if type(viewsIndices)==type(None):
        viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
    if not usedIndices:
        uesdIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
    NB_VIEW = len(viewsIndices)
    if weights==None:
        weights = np.array([1/NB_VIEW for i in range(NB_VIEW)])
    if sum(weights)!=1:
        weights = weights/sum(weights)
    monoviewData = np.concatenate([weights[index]*getV(DATASET, viewIndex, usedIndices)
                                        for index, viewIndex in enumerate(viewsIndices)], axis=1)
    return monoviewData


def gridSearch_hdf5(DATASET, viewsIndices, classificationKWARGS, learningIndices, metric=None, nIter=30):
    if type(viewsIndices)==type(None):
        viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
    fusionTypeName = classificationKWARGS["fusionType"]
    fusionTypePackage = globals()[fusionTypeName+"Package"]
    fusionMethodModuleName = classificationKWARGS["fusionMethod"]
    fusionMethodModule = getattr(fusionTypePackage, fusionMethodModuleName)
    classifiersNames = classificationKWARGS["classifiersNames"]
    bestSettings = []
    for classifierIndex, classifierName in enumerate(classifiersNames):
        logging.debug("\tStart:\t Random search for "+classifierName+ " with "+str(nIter)+" iterations")
        classifierModule = getattr(MonoviewClassifiers, classifierName)
        classifierMethod = getattr(classifierModule, "gridSearch")
        if fusionTypeName == "LateFusion":
            bestSettings.append(classifierMethod(getV(DATASET, viewsIndices[classifierIndex], learningIndices),
                                                 DATASET.get("Labels")[learningIndices], metric=metric,
                                                 nIter=nIter))
        else:
            bestSettings.append(classifierMethod(makeMonoviewData_hdf5(DATASET, usedIndices=learningIndices, viewsIndices=viewsIndices),
                                                 DATASET.get("Labels")[learningIndices], metric=metric,
                                                 nIter=nIter))
        logging.debug("\tDone:\t Random search for "+classifierName)
    classificationKWARGS["classifiersConfigs"] = bestSettings
    logging.debug("\tStart:\t Random search for "+fusionMethodModuleName)
    fusionMethodConfig = fusionMethodModule.gridSearch(DATASET, classificationKWARGS, learningIndices, nIter=nIter, viewsIndices=viewsIndices)
    logging.debug("\tDone:\t Random search for "+fusionMethodModuleName)
    return bestSettings, fusionMethodConfig


class Fusion:
    def __init__(self, NB_VIEW, DATASET_LENGTH, CLASS_LABELS, NB_CORES=1,**kwargs):
        fusionType = kwargs['fusionType']
        fusionMethod = kwargs['fusionMethod']
        fusionTypePackage = globals()[fusionType+"Package"]
        fusionMethodModule = getattr(fusionTypePackage, fusionMethod)
        fusionMethodClass = getattr(fusionMethodModule, fusionMethod)
        nbCores = NB_CORES
        classifierKWARGS = dict((key, value) for key, value in kwargs.iteritems() if key not in ['fusionType', 'fusionMethod'])
        print classifierKWARGS
        self.classifier = fusionMethodClass(NB_CORES=nbCores, **classifierKWARGS)

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):
        self.classifier.fit_hdf5(DATASET, trainIndices=trainIndices, viewsIndices=viewsIndices)

    def fit(self, DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, NB_CLASS, NB_CORES, trainArguments):
        fusionType, fusionMethod, fusionConfig, monoviewClassifier, monoviewClassifierConfig = trainArguments
        fusionTypeModule = globals()[fusionType]  # Early/late fusion
        trainFusion = getattr(fusionTypeModule, fusionMethod+"Train")  # linearWeighted for example
        classifier = trainFusion(DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, monoviewClassifier,
                                 monoviewClassifierConfig, fusionConfig)
        return fusionType, fusionMethod, classifier

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if usedIndices:
            predictedLabels = self.classifier.predict_hdf5(DATASET, usedIndices=usedIndices, viewsIndices=viewsIndices)
        else:
            predictedLabels = []
        return predictedLabels

    def predict_probas_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            predictedLabels = self.classifier.predict_probas_hdf5(DATASET, usedIndices=usedIndices)
        else:
            predictedLabels = []
        return predictedLabels

    def predict(self, DATASET, classifier, NB_CLASS):
        fusionType, fusionMethod, fusionClassifier = classifier
        fusionType = globals()[fusionType]  # Early/late fusion
        predictFusion = getattr(fusionType, fusionMethod+"Predict")  # linearWeighted for example
        predictedLabels = predictFusion(DATASET, fusionClassifier)
        return predictedLabels



