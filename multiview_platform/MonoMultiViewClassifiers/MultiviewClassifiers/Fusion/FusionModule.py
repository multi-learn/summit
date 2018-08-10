import logging
import pkgutil

import numpy as np

# from Methods import *

try:
    from . import Methods
except ValueError:
    import pdb;pdb.set_trace()

from ... import MonoviewClassifiers
from ...utils.Dataset import getV

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def genName(config):
    if config["fusionType"] == "LateFusion":
        classifierRedNames = [classifierName[:4] for classifierName in config["classifiersNames"]]
        return "Late-" + str(config["fusionMethod"][:4])#+"-"+"-".join(classifierRedNames)
    elif config["fusionType"] == "EarlyFusion":
        monoview_short_name = getattr(getattr(MonoviewClassifiers, config["classifiersNames"]),
                                         config["classifiersNames"])().get_name_for_fusion()
        return "Early-" + config["fusionMethod"][:4] + "-" + monoview_short_name


def getBenchmark(benchmark, args=None):
    """Used to generate the list of fusion classifiers for the benchmark"""
    fusionModulesNames = [name for _, name, isPackage
                          in pkgutil.iter_modules(['./MonoMultiViewClassifiers/MultiviewClassifiers/Fusion/Methods']) if not isPackage]
    fusionMethods = dict((fusionModulesName, [name for _, name, isPackage in
                                              pkgutil.iter_modules(
                                                  ["./MonoMultiViewClassifiers/MultiviewClassifiers/Fusion/Methods/" + fusionModulesName + "Package"])
                                              if not isPackage])
                         for fusionModulesName in fusionModulesNames)
    if args is None:
        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['./MonoMultiViewClassifiers/MonoviewClassifiers'])
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
                                                                                 "./MonoMultiViewClassifiers/MultiviewClassifiers/Fusion/Methods/LateFusionPackage"])
                                                                             if not isPackage]
            else:
                benchmark["Multiview"]["Fusion"]["Methods"]["LateFusion"] = args.FU_late_methods
        if "EarlyFusion" in benchmark["Multiview"]["Fusion"]["Methods"]:
            if args.FU_early_methods == [""]:
                benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = [name for _, name, isPackage in
                                                                              pkgutil.iter_modules([
                                                                                  "./MonoMultiViewClassifiers/MultiviewClassifiers/Fusion/Methods/EarlyFusionPackage"])
                                                                              if not isPackage]
            else:
                benchmark["Multiview"]["Fusion"]["Methods"]["EarlyFusion"] = args.FU_early_methods
        if args.CL_algos_monoview == ['']:
            benchmark["Multiview"]["Fusion"]["Classifiers"] = [name for _, name, isPackage in
                                                               pkgutil.iter_modules(['./MonoMultiViewClassifiers/MonoviewClassifiers'])
                                                               if (not isPackage) and (name != "SGD") and (
                                                                   name[:3] != "SVM")
                                                               and (name != "SCM")]
        else:
            benchmark["Multiview"]["Fusion"]["Classifiers"] = args.CL_algos_monoview
    return benchmark


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    """Used to generate the list of arguments for each fusion experimentation"""
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
    """Used to concatenate the viewsin one big monoview dataset"""
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
    """Used to generate parameters sets for the random hyper parameters optimization function"""
    fusionTypeName = classificationKWARGS["fusionType"]
    fusionTypePackage = getattr(Methods, fusionTypeName + "Package")
    fusionMethodModuleName = classificationKWARGS["fusionMethod"]
    fusionMethodModule = getattr(fusionTypePackage, fusionMethodModuleName)
    fusionMethodConfig = fusionMethodModule.genParamsSets(classificationKWARGS, randomState, nIter=nIter)
    return fusionMethodConfig


# def gridSearch_hdf5(DATASET, viewsIndices, classificationKWARGS, learningIndices, metric=None, nIter=30):
#     if type(viewsIndices) == type(None):
#         viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
#     fusionTypeName = classificationKWARGS["fusionType"]
#     fusionTypePackage = globals()[fusionTypeName + "Package"]
#     fusionMethodModuleName = classificationKWARGS["fusionMethod"]
#     fusionMethodModule = getattr(fusionTypePackage, fusionMethodModuleName)
#     classifiersNames = classificationKWARGS["classifiersNames"]
#     bestSettings = []
#     for classifierIndex, classifierName in enumerate(classifiersNames):
#         logging.debug("\tStart:\t Random search for " + classifierName + " with " + str(nIter) + " iterations")
#         classifierModule = getattr(MonoviewClassifiers, classifierName)
#         classifierMethod = getattr(classifierModule, "hyperParamSearch")
#         if fusionTypeName == "LateFusion":
#             bestSettings.append(classifierMethod(getV(DATASET, viewsIndices[classifierIndex], learningIndices),
#                                                  DATASET.get("Labels")[learningIndices], metric=metric,
#                                                  nIter=nIter))
#         else:
#             bestSettings.append(
#                 classifierMethod(makeMonoviewData_hdf5(DATASET, usedIndices=learningIndices, viewsIndices=viewsIndices),
#                                  DATASET.get("Labels")[learningIndices], metric=metric,
#                                  nIter=nIter))
#         logging.debug("\tDone:\t Random search for " + classifierName)
#     classificationKWARGS["classifiersConfigs"] = bestSettings
#     logging.debug("\tStart:\t Random search for " + fusionMethodModuleName)
#     fusionMethodConfig = fusionMethodModule.gridSearch(DATASET, classificationKWARGS, learningIndices, nIter=nIter,
#                                                        viewsIndices=viewsIndices)
#     logging.debug("\tDone:\t Random search for " + fusionMethodModuleName)
#     return bestSettings, fusionMethodConfig



class FusionClass:
    """The global representant of Fusion"""
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        fusionType = kwargs['fusionType']
        fusionMethod = kwargs['fusionMethod']
        fusionTypePackage = getattr(Methods, fusionType + "Package")
        fusionMethodModule = getattr(fusionTypePackage, fusionMethod)
        fusionMethodClass = getattr(fusionMethodModule, fusionMethod)
        nbCores = NB_CORES
        classifierKWARGS = dict(
            (key, value) for key, value in kwargs.items() if key not in ['fusionType', 'fusionMethod'])
        self.classifier = fusionMethodClass(randomState, NB_CORES=nbCores, **classifierKWARGS)

    def setParams(self, paramsSet):
        self.classifier.setParams(paramsSet)

    def fit_hdf5(self, DATASET, labels, trainIndices=None, viewsIndices=None, metric=["f1_score", None]):
        self.classifier.fit_hdf5(DATASET, labels, trainIndices=trainIndices, viewsIndices=viewsIndices)

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

    def getConfigString(self, classificationKWARGS):
        monoviewClassifiersNames = classificationKWARGS["classifiersNames"]
        monoviewClassifiersConfigs = classificationKWARGS["classifiersConfigs"]
        fusionMethodConfig = classificationKWARGS["fusionMethodConfig"]
        return self.classifier.getConfig(fusionMethodConfig, monoviewClassifiersNames,
                                                          monoviewClassifiersConfigs)

    def getSpecificAnalysis(self, classificationKWARGS):
        fusionType = classificationKWARGS["fusionType"]
        if fusionType == "LateFusion":
            stringAnalysis = Methods.LateFusion.getScores(self)
        else:
            stringAnalysis = ''
        return stringAnalysis
