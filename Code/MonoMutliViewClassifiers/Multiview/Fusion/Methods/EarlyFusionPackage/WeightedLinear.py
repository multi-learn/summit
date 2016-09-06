from ...Methods.EarlyFusion import EarlyFusionClassifier
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
            classifier = WeightedLinear(1, **classificationKWARGS)
            classifier.fit_hdf5(DATASET, trainIndices)
            predictedLabels = classifier.predict_hdf5(DATASET, trainIndices)
            accuracy = accuracy_score(DATASET.get("labels")[trainIndices], predictedLabels)
            if accuracy > bestScore:
                bestScore = accuracy
                bestConfig = normalizedArray
        return [bestConfig]


class WeightedLinear(EarlyFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        EarlyFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['classifiersConfigs'],
                                       NB_CORES=NB_CORES)
        self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))

    def fit_hdf5(self, DATASET, trainIndices=None):
        if not trainIndices:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        self.weights = self.weights/float(max(self.weights))
        self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=trainIndices)
        monoviewClassifierModule = getattr(MonoviewClassifiers, self.monoviewClassifierName)
        print self.monoviewClassifiersConfig
        self.monoviewClassifier = monoviewClassifierModule.fit(self.monoviewData, DATASET.get("labels")[trainIndices],
                                                                     NB_CORES=self.nbCores, **self.monoviewClassifiersConfig)
                                                                     #**dict((str(configIndex), config) for configIndex, config in
                                                                      #      enumerate(self.monoviewClassifiersConfig
                                                                       #               )))

    def predict_hdf5(self, DATASET, usedIndices=None):
        self.weights = self.weights/float(max(self.weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=usedIndices)
            predictedLabels = self.monoviewClassifier.predict(self.monoviewData)
        else:
            predictedLabels=[]
        return predictedLabels

    def predict_proba_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            self.makeMonoviewData_hdf5(DATASET, weights=self.weights, usedIndices=usedIndices)
            predictedLabels = self.monoviewClassifier.predict_proba(self.monoviewData)
        else:
            predictedLabels=[]
        return predictedLabels

    def getConfig(self, fusionMethodConfig ,monoviewClassifiersNames, monoviewClassifiersConfigs):
        configString = "with weighted concatenation, using weights : "+", ".join(map(str, self.weights))+ \
                       " with monoview classifier : "
        monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifiersNames[0])
        configString += monoviewClassifierModule.getConfig(monoviewClassifiersConfigs[0])
        return configString

    def gridSearch(self, classificationKWARGS):

        return