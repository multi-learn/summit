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


def gridSearch(DATASET, classificationKWARGS, trainIndices, nIter=30, viewsIndices=None):
    if type(viewsIndices)==type(None):
        viewsIndices = np.arange(DATASET.get("Metadata").attrs["nbView"])
    nbView = len(viewsIndices)
    bestScore = 0.0
    bestConfig = None
    if classificationKWARGS["fusionMethodConfig"][0] is not None:
        for i in range(nIter):
            randomWeightsArray = np.random.random_sample(nbView)
            normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
            classificationKWARGS["fusionMethodConfig"][0] = normalizedArray
            classifier = WeightedLinear(1, **classificationKWARGS)
            classifier.fit_hdf5(DATASET, trainIndices, viewsIndices=viewsIndices)
            predictedLabels = classifier.predict_hdf5(DATASET, trainIndices, viewsIndices=viewsIndices)
            accuracy = accuracy_score(DATASET.get("Labels")[trainIndices], predictedLabels)
            if accuracy > bestScore:
                bestScore = accuracy
                bestConfig = normalizedArray
        return [bestConfig]


class WeightedLinear(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                      NB_CORES=NB_CORES)
        if kwargs['fusionMethodConfig'][0]==None:
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
            predictedLabels = []
            viewScores = np.zeros((nbView, len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
            for index, viewIndex in enumerate(viewsIndices):
                viewScores[index] = self.monoviewClassifiers[index].predict_proba(
                    getV(DATASET, viewIndex, usedIndices))
            for currentIndex, usedIndex in enumerate(usedIndices):
                predictedLabel = np.argmax(np.array(
                    [max(viewScore) * weight for viewScore, weight in zip(viewScores[:, currentIndex], self.weights)],
                    dtype=float))
                predictedLabels.append(predictedLabel)
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
