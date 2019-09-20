import numpy as np

from ...Methods.EarlyFusion import EarlyFusionClassifier
from ..... import monoview_classifiers


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    nbView = classificationKWARGS["nbView"]
    if classificationKWARGS["classifiersConfigs"] is None:
        monoviewClassifierModule = getattr(monoview_classifiers, classificationKWARGS["classifiersNames"])
        paramsMonoview = monoviewClassifierModule.paramsToSet(nIter, randomState)
    paramsSets = []
    for iterIndex in range(nIter):
        randomWeightsArray = randomState.random_sample(nbView)
        normalizedArray = randomWeightsArray / np.sum(randomWeightsArray)
        paramsSets.append([normalizedArray, paramsMonoview[iterIndex]])
    return paramsSets


def getArgs(benchmark, args, views, viewsIndices, directory, resultsMonoview, classificationIndices):
    argumentsList = []
    if args.FU_E_cl_names != ['']:
        pass
    else:
        monoviewClassifierModulesNames = benchmark["monoview"]
        args.FU_E_cl_names = monoviewClassifierModulesNames
        args.FU_E_cl_config = [None for _ in monoviewClassifierModulesNames]
    for classifierName, classifierConfig in zip(args.FU_E_cl_names, args.FU_E_cl_config):
        monoviewClassifierModule = getattr(monoview_classifiers, classifierName)
        if classifierConfig is not None:
            arguments = {"CL_type": "fusion",
                         "views": views,
                         "NB_VIEW": len(views),
                         "viewsIndices": viewsIndices,
                         "NB_CLASS": len(args.CL_classes),
                         "LABELS_NAMES": args.CL_classes,
                         "FusionKWARGS": {"fusionType": "EarlyFusion",
                                          "fusionMethod": "WeightedLinear",
                                          "classifiersNames": classifierName,
                                          "classifiersConfigs": monoviewClassifierModule.getKWARGS([arg.split(":")
                                                                                                    for arg in
                                                                                                    classifierConfig.split(
                                                                                                        ",")]),
                                          'fusionMethodConfig': args.FU_E_method_configs,
                                          "nbView": (len(viewsIndices))}}
        else:
            arguments = {"CL_type": "fusion",
                         "views": views,
                         "NB_VIEW": len(views),
                         "viewsIndices": viewsIndices,
                         "NB_CLASS": len(args.CL_classes),
                         "LABELS_NAMES": args.CL_classes,
                         "FusionKWARGS": {"fusionType": "EarlyFusion",
                                          "fusionMethod": "WeightedLinear",
                                          "classifiersNames": classifierName,
                                          "classifiersConfigs": None,
                                          'fusionMethodConfig': args.FU_E_method_configs,
                                          "nbView": (len(viewsIndices))}}
        argumentsList.append(arguments)
    return argumentsList


class WeightedLinear(EarlyFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        EarlyFusionClassifier.__init__(self, randomState, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                       NB_CORES=NB_CORES)
        if kwargs['fusionMethodConfig'] is None:
            self.weights = np.ones(len(kwargs["classifiersNames"]), dtype=float)
        elif kwargs['fusionMethodConfig'] == ['']:
            self.weights = np.ones(len(kwargs["classifiersNames"]), dtype=float)
        else:
            self.weights = np.array(map(float, kwargs['fusionMethodConfig']))
        self.weights /= float(max(self.weights))



    def fit_hdf5(self, DATASET, labels, trainIndices=None, viewsIndices=None):
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        if trainIndices is None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        self.weights /= float(max(self.weights))
        self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=trainIndices, viewsIndices=viewsIndices)
        # monoviewClassifierModule = getattr(monoview_classifiers, self.monoviewClassifierName)
        self.monoviewClassifier.fit(self.monoviewData, labels[trainIndices])
                                                               # self.randomState,
                                                               # NB_CORES=self.nbCores,
                                                               # **self.monoviewClassifiersConfig)

    def setParams(self, paramsSet):
        self.weights = paramsSet[0]
        self.monoviewClassifiersConfig = paramsSet[1]

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices) == type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        self.weights /= float(np.sum(self.weights))
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=usedIndices, viewsIndices=viewsIndices)
        predictedLabels = self.monoviewClassifier.predict(self.monoviewData)

        return predictedLabels

    def predict_proba_hdf5(self, DATASET, usedIndices=None):
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=usedIndices)
        predictedLabels = self.monoviewClassifier.predict_proba(self.monoviewData)
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames, monoviewClassifiersConfigs):
        configString = "with weighted concatenation, using weights : " + ", ".join(map(str, self.weights)) + \
                       " with monoview classifier : "
        # monoviewClassifierModule = getattr(monoview_classifiers, monoviewClassifiersNames)
        configString += self.monoviewClassifier.getConfig()
        return configString

    def gridSearch(self, classificationKWARGS):

        return
