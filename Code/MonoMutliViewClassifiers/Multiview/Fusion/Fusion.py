import numpy as np
import logging
import pkgutil

import Methods
from ... import MonoviewClassifiers
from ...utils.Dataset import getV

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def getBenchmark(benchmark, args=None):
    fusionModulesNames = [name for _, name, isPackage
                          in pkgutil.iter_modules(['Multiview/Fusion/Methods']) if not isPackage]
    fusionModules = [getattr(Methods, fusionModulesName)
                     for fusionModulesName in fusionModulesNames]
    fusionClassifiers = [getattr(fusionModule, fusionModulesName + "Classifier")
                         for fusionModulesName, fusionModule in zip(fusionModulesNames, fusionModules)]
    fusionMethods = dict((fusionModulesName, [name for _, name, isPackage in
                                              pkgutil.iter_modules(
                                                  ["Multiview/Fusion/Methods/" + fusionModulesName + "Package"])
                                              if not isPackage])
                         for fusionModulesName, fusionClasse in zip(fusionModulesNames, fusionClassifiers))
    if args is None:
        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['MonoviewClassifiers'])
                            if (not isPackage)]
        fusionMonoviewClassifiers = allMonoviewAlgos
        allFusionAlgos = {"Methods": fusionMethods, "Classifiers": fusionMonoviewClassifiers}
        benchmark["Multiview"]["Fusion"] = allFusionAlgos
    else:
        benchmark["Multiview"]["Fusion"] = {}
        if args.FU_types != [""]:
            benchmark["Multiview"]["Fusion"]["Methods"] = dict(
                (fusionType, []) for fusionType in args.FU_types)
        else:
            benchmark["Multiview"]["Fusion"]["Methods"] = dict(
                (fusionModulesName, "_") for fusionModulesName in fusionModulesNames)
        if "LateFusion" in benchmark["Multiview"]["Fusion"]["Methods"]:
            if args.FU_late_methods == [""]:
                benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = [name for _, name, isPackage in
                                                                             pkgutil.iter_modules([
                                                                                 "Multiview/Fusion/Methods/LateFusionPackage"])
                                                                             if not isPackage]
            else:
                benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = args.FU_late_methods
        if "EarlyFusion" in benchmark["Multiview"]["Fusion"]["Methods"]:
            if args.FU_early_methods == [""]:
                benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = [name for _, name, isPackage in
                                                                              pkgutil.iter_modules([
                                                                                  "Multiview/Fusion/Methods/EarlyFusionPackage"])
                                                                              if not isPackage]
            else:
                benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = args.FU_early_methods
        if args.CL_algos_monoview == ['']:
            benchmark["Multiview"]["Fusion"]["Classifiers"] = [name for _, name, isPackage in
                                                               pkgutil.iter_modules(['MonoviewClassifiers'])
                                                               if (not isPackage) and (name != "SGD") and (
                                                                   name[:3] != "SVM")
                                                               and (name != "SCM")]
        else:
            benchmark["Multiview"]["Fusion"]["Classifiers"] = args.CL_algos_monoview
    return benchmark


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    if not "Monoview" in benchmark and not args.FU_L_select_monoview in ["randomClf", "Determined"]:
        args.FU_L_select_monoview = "randomClf"
    argumentsList = []
    for fusionType in benchmark["Multiview"]["Fusion"]["Methods"]:
        fusionTypePackage = getattr(Methods, fusionType + "Package")
        for fusionMethod in benchmark["Multiview"]["Fusion"]["Methods"][fusionType]:
            fusionMethodModule = getattr(fusionTypePackage, fusionMethod)
            arguments = fusionMethodModule.getArgs(benchmark, args, views, viewsIndices, directory, resultsMonoview,
                                                   classificationIndices)
            argumentsList += arguments
    return argumentsList


def makeMonoviewData_hdf5(DATASET, weights=None, usedIndices=None, viewsIndices=None):
    if type(viewsIndices) == type(None):
        viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
    if not usedIndices:
        usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
    NB_VIEW = len(viewsIndices)
    if weights is None:
        weights = np.array([1 / NB_VIEW for i in range(NB_VIEW)])
    if sum(weights) != 1:
        weights = weights / sum(weights)
    monoviewData = np.concatenate([weights[index] * getV(DATASET, viewIndex, usedIndices)
                                   for index, viewIndex in enumerate(viewsIndices)], axis=1)
    return monoviewData


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    fusionTypeName = classificationKWARGS["fusionType"]
    fusionTypePackage = getattr(Methods, fusionTypeName + "Package")
    fusionMethodModuleName = classificationKWARGS["fusionMethod"]
    fusionMethodModule = getattr(fusionTypePackage, fusionMethodModuleName)
    fusionMethodConfig = fusionMethodModule.genParamsSets(classificationKWARGS, randomState, nIter=nIter)
    return fusionMethodConfig


def gridSearch_hdf5(DATASET, viewsIndices, classificationKWARGS, learningIndices, metric=None, nIter=30):
    if type(viewsIndices) == type(None):
        viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
    fusionTypeName = classificationKWARGS["fusionType"]
    fusionTypePackage = globals()[fusionTypeName + "Package"]
    fusionMethodModuleName = classificationKWARGS["fusionMethod"]
    fusionMethodModule = getattr(fusionTypePackage, fusionMethodModuleName)
    classifiersNames = classificationKWARGS["classifiersNames"]
    bestSettings = []
    for classifierIndex, classifierName in enumerate(classifiersNames):
        logging.debug("\tStart:\t Random search for " + classifierName + " with " + str(nIter) + " iterations")
        classifierModule = getattr(MonoviewClassifiers, classifierName)
        classifierMethod = getattr(classifierModule, "hyperParamSearch")
        if fusionTypeName == "LateFusion":
            bestSettings.append(classifierMethod(getV(DATASET, viewsIndices[classifierIndex], learningIndices),
                                                 DATASET.get("Labels")[learningIndices], metric=metric,
                                                 nIter=nIter))
        else:
            bestSettings.append(
                classifierMethod(makeMonoviewData_hdf5(DATASET, usedIndices=learningIndices, viewsIndices=viewsIndices),
                                 DATASET.get("Labels")[learningIndices], metric=metric,
                                 nIter=nIter))
        logging.debug("\tDone:\t Random search for " + classifierName)
    classificationKWARGS["classifiersConfigs"] = bestSettings
    logging.debug("\tStart:\t Random search for " + fusionMethodModuleName)
    fusionMethodConfig = fusionMethodModule.gridSearch(DATASET, classificationKWARGS, learningIndices, nIter=nIter,
                                                       viewsIndices=viewsIndices)
    logging.debug("\tDone:\t Random search for " + fusionMethodModuleName)
    return bestSettings, fusionMethodConfig


def getCLString(classificationKWARGS):
    if classificationKWARGS["fusionType"] == "LateFusion":
        return "Fusion-" + classificationKWARGS["fusionType"] + "-" + classificationKWARGS["fusionMethod"] + "-" + \
               "-".join(classificationKWARGS["classifiersNames"])
    elif classificationKWARGS["fusionType"] == "EarlyFusion":
        return "Fusion-" + classificationKWARGS["fusionType"] + "-" + classificationKWARGS["fusionMethod"] + "-" + \
               classificationKWARGS["classifiersNames"]


class Fusion:
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        fusionType = kwargs['fusionType']
        fusionMethod = kwargs['fusionMethod']
        fusionTypePackage = getattr(Methods, fusionType + "Package")
        fusionMethodModule = getattr(fusionTypePackage, fusionMethod)
        fusionMethodClass = getattr(fusionMethodModule, fusionMethod)
        nbCores = NB_CORES
        classifierKWARGS = dict(
            (key, value) for key, value in kwargs.iteritems() if key not in ['fusionType', 'fusionMethod'])
        self.classifier = fusionMethodClass(randomState, NB_CORES=nbCores, **classifierKWARGS)

    def setParams(self, paramsSet):
        self.classifier.setParams(paramsSet)

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None, metric=["f1_score", None]):
        self.classifier.fit_hdf5(DATASET, trainIndices=trainIndices, viewsIndices=viewsIndices)

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        predictedLabels = self.classifier.predict_hdf5(DATASET, usedIndices=usedIndices, viewsIndices=viewsIndices)
        return predictedLabels

    def predict_probas_hdf5(self, DATASET, usedIndices=None):
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            predictedLabels = self.classifier.predict_probas_hdf5(DATASET, usedIndices=usedIndices)
        else:
            predictedLabels = []
        return predictedLabels
