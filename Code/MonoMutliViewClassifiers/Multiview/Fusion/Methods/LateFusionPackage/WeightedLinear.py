from ...Methods.LateFusion import LateFusionClassifier
import MonoviewClassifiers
import numpy as np
from sklearn.metrics import accuracy_score
from utils.Dataset import getV


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    nbView = classificationKWARGS["nbView"]
    paramsSets = []
    for _ in range(nIter):
        randomWeightsArray = randomState.random_sample(nbView)
        normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
        paramsSets.append([normalizedArray])
    return paramsSets


def getArgs(args, views, viewsIndices, directory):
    monoviewClassifierModules = [getattr(MonoviewClassifiers, classifierName) for classifierName in args.FU_L_cl_names]
    arguments = {"CL_type": "Fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion",
                                  "fusionMethod": "BayesianInference",
                                  "classifiersNames": args.FU_L_cl_names,
                                  "classifiersConfigs": [monoviewClassifierModule.getKWARGS([arg.split(":")
                                                                                             for arg in
                                                                                             classifierConfig.split(";")])
                                                         for monoviewClassifierModule,classifierConfig
                                                         in zip(args.FU_L_cl_config,monoviewClassifierModules)],
                                  'fusionMethodConfig': args.FU_L_method_config[0],
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]


class WeightedLinear(LateFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, randomState, kwargs['classifiersNames'], kwargs['classifiersConfigs'], kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)
        if kwargs['fusionMethodConfig'][0]==None or kwargs['fusionMethodConfig'][0]==['']:
            self.weights = np.ones(len(kwargs["classifiersNames"]), dtype=float)
        else:
            self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))
        self.needProbas = True

    def setParams(self, paramsSet):
        self.weights = paramsSet[0]

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        self.weights = self.weights/float(sum(self.weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            viewScores = np.zeros((nbView, len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
            for index, viewIndex in enumerate(viewsIndices):
                viewScores[index] = np.array(self.monoviewClassifiers[index].predict_proba(
                    getV(DATASET, viewIndex, usedIndices)))*self.weights[index]
            predictedLabels = np.argmax(np.sum(viewScores, axis=0), axis=1)
        else:
            predictedLabels = []

        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Weighted linear using a weight for each view : "+", ".join(map(str,self.weights)) + \
                       "\n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString
