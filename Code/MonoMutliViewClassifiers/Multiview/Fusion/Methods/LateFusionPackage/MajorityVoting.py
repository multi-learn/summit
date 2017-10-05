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
                                                                                             classifierConfig.split(";")])
                                                         for monoviewClassifierModule,classifierConfig
                                                         in zip(args.FU_L_cl_config,monoviewClassifierModules)],
                                  'fusionMethodConfig': args.FU_L_method_config[0],
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]

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
#             classifier = MajorityVoting(1, **classificationKWARGS)
#             classifier.fit_hdf5(DATASET, trainIndices, viewsIndices=viewsIndices)
#             predictedLabels = classifier.predict_hdf5(DATASET, trainIndices, viewsIndices=viewsIndices)
#             accuracy = accuracy_score(DATASET.get("Labels")[trainIndices], predictedLabels)
#             if accuracy > bestScore:
#                 bestScore = accuracy
#                 bestConfig = normalizedArray
#         return [bestConfig]


class MajorityVoting(LateFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, randomState, kwargs['classifiersNames'], kwargs['classifiersConfigs'], kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)
        if kwargs['fusionMethodConfig'][0]==None or kwargs['fusionMethodConfig'][0]==['']:
            self.weights = np.ones(len(kwargs["classifiersNames"]), dtype=float)
        else:
            self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))

    def setParams(self, paramsSet):
        self.weights = paramsSet[0]

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        self.weights = self.weights/float(max(self.weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            datasetLength = len(usedIndices)
            votes = np.zeros((datasetLength, DATASET.get("Metadata").attrs["nbClass"]), dtype=int)
            monoViewDecisions = np.zeros((len(usedIndices),nbView), dtype=int)
            for index, viewIndex in enumerate(viewsIndices):
                monoViewDecisions[:, index] = self.monoviewClassifiers[index].predict(
                    getV(DATASET, viewIndex, usedIndices))
            for exampleIndex in range(datasetLength):
                for viewIndex, featureClassification in enumerate(monoViewDecisions[exampleIndex, :]):
                    votes[exampleIndex, featureClassification] += self.weights[viewIndex]
                nbMaximum = len(np.where(votes[exampleIndex] == max(votes[exampleIndex]))[0])
                try:
                    assert nbMaximum != nbView
                except:
                    print "Majority voting can't decide, each classifier has voted for a different class"
                    raise
            predictedLabels = np.argmax(votes, axis=1)
            # Can be upgraded by restarting a new classification process if
            # there are multiple maximums ?:
            # 	while nbMaximum>1:
            # 		relearn with only the classes that have a maximum number of vote
            # 		votes = revote
            # 		nbMaximum = len(np.where(votes==max(votes))[0])
        else:
            predictedLabels = []
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Majority Voting \n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString