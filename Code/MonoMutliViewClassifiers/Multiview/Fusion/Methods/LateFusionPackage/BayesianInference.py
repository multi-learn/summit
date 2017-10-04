from ...Methods.LateFusion import LateFusionClassifier
import MonoviewClassifiers
import numpy as np
from sklearn.metrics import accuracy_score
from utils.Dataset import getV


def genParamsSets(classificationKWARGS, nIter=1):
    nbView = classificationKWARGS["nbView"]
    paramsSets = []
    for _ in range(nIter):
        randomWeightsArray = np.random.random_sample(nbView)
        normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
        paramsSets.append([normalizedArray])
    return paramsSets


# def getArgs(args, benchmark):
#     classifiersNames = args.FU_cl_names
#     classifiersConfig = [getattr(MonoviewClassifiers, name).getKWARGS([arg.split(":")
#                                                                        for arg in config.split(";")])
#                          for config, name in zip(args.FU_cl_config, classifiersNames)]
#     fusionMethodConfig = args.FU_method_config
#     return classifiersNames, classifiersConfig, fusionMethodConfig

def getArgs(args, views, viewsIndices):
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
                                                                                            classifierConfig.split(",")])
                                                         for monoviewClassifierModule,classifierConfig
                                                         in zip(monoviewClassifierModules,args.FU_L_cl_config)],
                                  'fusionMethodConfig': args.FU_L_method_config,
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]
#
# def gridSearch(DATASET, classificationKWARGS, trainIndices, nIter=30, viewsIndices=None):
#     if type(viewsIndices)==type(None):
#         viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
#     nbView = len(viewsIndices)
#     bestScore = 0.0
#     bestConfig = None
#     if classificationKWARGS["fusionMethodConfig"][0] is not None:
#         for i in range(nIter):
#             randomWeightsArray = np.random.random_sample(nbView)
#             normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
#             classificationKWARGS["fusionMethodConfig"][0] = normalizedArray
#             classifier = BayesianInference(1, **classificationKWARGS)
#             classifier.fit_hdf5(DATASET, trainIndices, viewsIndices=viewsIndices)
#             predictedLabels = classifier.predict_hdf5(DATASET, trainIndices, viewsIndices=viewsIndices)
#             accuracy = accuracy_score(DATASET.get("Labels")[trainIndices], predictedLabels)
#             if accuracy > bestScore:
#                 bestScore = accuracy
#                 bestConfig = normalizedArray
#         return [bestConfig]


class BayesianInference(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['classifiersConfigs'], kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)

        # self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))
        if kwargs['fusionMethodConfig'][0]==None or kwargs['fusionMethodConfig']==['']:
            self.weights = [1.0 for classifier in kwargs['classifiersNames']]
        else:
            self.weights = np.array(map(float, kwargs['fusionMethodConfig']))
        self.needProbas = True

    def setParams(self, paramsSet):
        self.weights = paramsSet[0]

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        self.weights = self.weights/float(max(self.weights))
        nbView = len(viewsIndices)
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if sum(self.weights)!=1.0:
            self.weights = self.weights/sum(self.weights)
        if usedIndices:

            viewScores = np.zeros((nbView, len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
            for index, viewIndex in enumerate(viewsIndices):
                viewScores[index] = np.power(self.monoviewClassifiers[index].predict_proba(getV(DATASET, viewIndex, usedIndices)),
                                                 self.weights[index])
            predictedLabels = np.argmax(np.prod(viewScores, axis=0), axis=1)
        else:
            predictedLabels = []
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Bayesian Inference using a weight for each view : "+", ".join(map(str, self.weights)) + \
                       "\n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString