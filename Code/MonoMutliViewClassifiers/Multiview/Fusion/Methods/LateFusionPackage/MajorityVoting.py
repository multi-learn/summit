from ..LateFusion import LateFusionClassifier, getClassifiers, getConfig
import MonoviewClassifiers
import numpy as np
from sklearn.metrics import accuracy_score
from utils.Dataset import getV
import pkgutil


def genParamsSets(classificationKWARGS, randomState, nIter=1):
    nbView = classificationKWARGS["nbView"]
    paramsSets = []
    for _ in range(nIter):
        randomWeightsArray = randomState.random_sample(nbView)
        normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
        paramsSets.append([normalizedArray])
    return paramsSets


def getArgs(benchmark, args, views, viewsIndices, directory, resultsMonoview, classificationIndices):
    if args.FU_L_cl_names!=['']:
        pass
    else:
        monoviewClassifierModulesNames = benchmark["Monoview"]
        args.FU_L_cl_names = getClassifiers(args.FU_L_select_monoview, monoviewClassifierModulesNames, directory, viewsIndices, resultsMonoview, classificationIndices)
    monoviewClassifierModules = [getattr(MonoviewClassifiers, classifierName)
                                 for classifierName in args.FU_L_cl_names]
    if args.FU_L_cl_config != ['']:
        classifiersConfigs = [monoviewClassifierModule.getKWARGS([arg.split(":") for arg in classifierConfig.split(",")])
                              for monoviewClassifierModule,classifierConfig
                              in zip(monoviewClassifierModules,args.FU_L_cl_config)]
    else:
        classifiersConfigs = getConfig(args.FU_L_cl_names, resultsMonoview)
    arguments = {"CL_type": "Fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion",
                                  "fusionMethod": "MajorityVoting",
                                  "classifiersNames": args.FU_L_cl_names,
                                  "classifiersConfigs": classifiersConfigs,
                                  'fusionMethodConfig': args.FU_L_method_config,
                                  'monoviewSelection': args.FU_L_select_monoview,
                                  "nbView": (len(viewsIndices))}}
    return [arguments]


class MajorityVoting(LateFusionClassifier):
    def __init__(self, randomState, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, randomState, kwargs['classifiersNames'], kwargs['classifiersConfigs'], kwargs["monoviewSelection"],
                                      NB_CORES=NB_CORES)
        if kwargs['fusionMethodConfig'][0] is None or kwargs['fusionMethodConfig']==['']:
            self.weights = np.ones(len(kwargs["classifiersNames"]), dtype=float)
        else:
            self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))

    def setParams(self, paramsSet):
        self.weights = np.array(paramsSet[0])

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        if type(viewsIndices)==type(None):
            viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
        nbView = len(viewsIndices)
        self.weights /= float(sum(self.weights))
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])

        datasetLength = len(usedIndices)
        votes = np.zeros((datasetLength, DATASET.get("Metadata").attrs["nbClass"]), dtype=float)
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
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Majority Voting \n\t-With weights : "+str(self.weights)+"\n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString