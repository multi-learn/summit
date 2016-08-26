from LateFusion import LateFusionClassifier
import MonoviewClassifiers
import numpy as np
from sklearn.metrics import accuracy_score


def gridSearch(DATASET, classificationKWARGS, trainIndices):
    bestScore = 0.0
    bestConfig = None
    if classificationKWARGS["fusionMethodConfig"][0] is not None:
        for i in range(0):
            randomWeightsArray = np.random.random_sample(len(DATASET.get("Metadata").attrs["nbView"]))
            normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
            classificationKWARGS["fusionMethodConfig"][0] = normalizedArray
            classifier = WeightedLinear(1, **classificationKWARGS)
            classifier.fit_hdf5(DATASET, trainIndices)
            predictedLabels = classifier.predict_hdf5(DATASET, trainIndices)
            accuracy = accuracy_score(DATASET.get("labels")[trainIndices], predictedLabels)
            if accuracy > bestScore:
                bestScore = accuracy
                bestConfig = normalizedArray
        return bestConfig


class WeightedLinear(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['monoviewClassifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.weights = map(float, kwargs['fusionMethodConfig'][0])

    def predict_hdf5(self, DATASET, usedIndices=None):
        # Normalize weights ?
        # weights = weights/float(max(weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            predictedLabels = []
            viewScores = np.zeros((DATASET.get("Metadata").attrs["nbView"], len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
            for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
                viewScores[viewIndex] = self.monoviewClassifiers[viewIndex].predict_proba(
                    DATASET.get("View" + str(viewIndex))[usedIndices])
            for currentIndex, usedIndex in enumerate(usedIndices):
                predictedLabel = np.argmax(np.array(
                    [max(viewScore) * weight for viewScore, weight in zip(viewScores[:, currentIndex], self.weights)],
                    dtype=float))
                predictedLabels.append(predictedLabel)
                # fusedExamples = np.array([sum(np.array([featureScores * weight for weight, featureScores in zip(weights, exampleDecisions)])) for exampleDecisions in monoViewDecisions])
        else:
            predictedLabels = []

        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Weighted linear using a weight for each view : "+", ".join(self.weights) + \
                       "\n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString
