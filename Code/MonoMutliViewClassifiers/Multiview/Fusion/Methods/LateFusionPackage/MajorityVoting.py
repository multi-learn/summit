from ...Methods.LateFusion import LateFusionClassifier
import MonoviewClassifiers
import numpy as np
from sklearn.metrics import accuracy_score


def gridSearch(DATASET, classificationKWARGS, trainIndices, nIter=30):
    bestScore = 0.0
    bestConfig = None
    if classificationKWARGS["fusionMethodConfig"][0] is not None:
        for i in range(nIter):
            randomWeightsArray = np.random.random_sample(DATASET.get("Metadata").attrs["nbView"])
            normalizedArray = randomWeightsArray/np.sum(randomWeightsArray)
            classificationKWARGS["fusionMethodConfig"][0] = normalizedArray
            classifier = MajorityVoting(1, **classificationKWARGS)
            classifier.fit_hdf5(DATASET, trainIndices)
            predictedLabels = classifier.predict_hdf5(DATASET, trainIndices)
            accuracy = accuracy_score(DATASET.get("labels")[trainIndices], predictedLabels)
            if accuracy > bestScore:
                bestScore = accuracy
                bestConfig = normalizedArray
        return [bestConfig]


class MajorityVoting(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))

    def predict_hdf5(self, DATASET, usedIndices=None):
        self.weights = self.weights/float(max(self.weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            datasetLength = len(usedIndices)
            votes = np.zeros((datasetLength, DATASET.get("Metadata").attrs["nbClass"]), dtype=int)
            monoViewDecisions = np.zeros((len(usedIndices),DATASET.get("Metadata").attrs["nbView"]), dtype=int)
            for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
                monoViewDecisions[:, viewIndex] = self.monoviewClassifiers[viewIndex].predict(
                    DATASET.get("View" + str(viewIndex))[usedIndices])
            for exampleIndex in range(datasetLength):
                for viewIndex, featureClassification in enumerate(monoViewDecisions[exampleIndex, :]):
                    votes[exampleIndex, featureClassification] += self.weights[viewIndex]
                nbMaximum = len(np.where(votes[exampleIndex] == max(votes[exampleIndex]))[0])
                try:
                    assert nbMaximum != DATASET.get("Metadata").attrs["nbView"]
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