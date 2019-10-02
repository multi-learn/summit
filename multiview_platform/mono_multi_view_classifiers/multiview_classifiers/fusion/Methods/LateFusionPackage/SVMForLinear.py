import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import pkgutil

from ..LateFusion import LateFusionClassifier, getClassifiers, getConfig
from .....utils.dataset import get_v
from ..... import monoview_classifiers


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    paramsSets = []
    for _ in range(nIter):
        paramsSets.append([])
    return paramsSets


def getArgs(benchmark, args, views, viewsIndices, directory, resultsMonoview, classificationIndices):
    if args.FU_L_cl_names != ['']:
        pass
    else:
        monoviewClassifierModulesNames = benchmark["monoview"]
        args.FU_L_cl_names = getClassifiers(args.FU_L_select_monoview, monoviewClassifierModulesNames, directory,
                                            viewsIndices, resultsMonoview, classificationIndices)
    monoviewClassifierModules = [getattr(monoview_classifiers, classifierName)
                                 for classifierName in args.FU_L_cl_names]
    if args.FU_L_cl_names == [""] and args.CL_type == ["multiview"]:
        raise AttributeError("You must perform monoview classification or specify "
                             "which monoview classifier to use Late fusion")
    if args.FU_L_cl_config != ['']:
        classifiersConfigs = [
            monoviewClassifierModule.getKWARGS([arg.split(":") for arg in classifierConfig.split(",")])
            for monoviewClassifierModule, classifierConfig
            in zip(monoviewClassifierModules, args.FU_L_cl_config)]
    else:
        classifiersConfigs = getConfig(args.FU_L_cl_names, resultsMonoview, viewsIndices)
    arguments = {"CL_type": "fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion",
                                  "fusionMethod": "SVMForLinear",
                                  "classifiers_names": args.FU_L_cl_names,
                                  "classifiersConfigs": classifiersConfigs,
                                  'fusionMethodConfig': args.FU_L_method_config,
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]


class SVMForLinear(LateFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, randomState, kwargs['classifiers_names'], kwargs['classifiersConfigs'],
                                      kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)
        self.SVMClassifier = None

    def fit_hdf5(self, DATASET, labels, trainIndices=None, viewsIndices=None):
        if viewsIndices is None:
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices is None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        for index, viewIndex in enumerate(viewsIndices):
            self.monoviewClassifiers[index].fit(get_v(DATASET, viewIndex, trainIndices),
                                       labels[trainIndices])
        self.SVMForLinearFusionFit(DATASET, labels, usedIndices=trainIndices, viewsIndices=viewsIndices)

    def setParams(self, paramsSet):
        pass

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if viewsIndices is None:
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        monoviewDecisions = np.zeros((len(usedIndices), nbView), dtype=int)
        for index, viewIndex in enumerate(viewsIndices):
            monoviewDecisions[:, index] = self.monoviewClassifiers[index].predict(
                get_v(DATASET, viewIndex, usedIndices))
        predictedLabels = self.SVMClassifier.predict(monoviewDecisions)
        return predictedLabels

    def SVMForLinearFusionFit(self, DATASET, labels, usedIndices=None, viewsIndices=None):
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        self.SVMClassifier = OneVsOneClassifier(SVC())
        monoViewDecisions = np.zeros((len(usedIndices), nbView), dtype=int)
        for index, viewIndex in enumerate(viewsIndices):
            monoViewDecisions[:, index] = self.monoviewClassifiers[index].predict(
                get_v(DATASET, viewIndex, usedIndices))

        self.SVMClassifier.fit(monoViewDecisions, labels[usedIndices])

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames, monoviewClassifiersConfigs):
        configString = "with SVM for linear \n\t-With monoview classifiers : "
        for monoviewClassifier in self.monoviewClassifiers:

            configString += monoviewClassifier.getConfig()
        return configString
