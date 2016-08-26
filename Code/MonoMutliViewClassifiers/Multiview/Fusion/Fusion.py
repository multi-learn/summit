from Methods import *


def gridSearch_hdf5(DATASET, classifiersNames):
    bestSettings = []
    for classifierIndex, classifierName in enumerate(classifiersNames):
        classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
        classifierMethod = getattr(classifierModule, "gridSearch")
        bestSettings.append(classifierMethod(DATASET.get("View"+str(classifierIndex))[...],
                                             DATASET.get("labels")[...]))
    return bestSettings


class Fusion:
    def __init__(self, NB_VIEW, DATASET_LENGTH, CLASS_LABELS, NB_CORES=1,**kwargs):
        fusionType = kwargs['fusionType']
        fusionMethod = kwargs['fusionMethod']
        fusionTypeModule = globals()[fusionType]
        fusionMethodClass = getattr(fusionTypeModule, fusionMethod)
        nbCores = NB_CORES
        classifierKWARGS = dict((key, value) for key, value in kwargs.iteritems() if key not in ['fusionType', 'fusionMethod'])
        self.classifier = fusionMethodClass(NB_CORES=nbCores, **classifierKWARGS)

    def fit_hdf5(self, DATASET, trainIndices=None):
        self.classifier.fit_hdf5(DATASET, trainIndices=trainIndices)

    def fit(self, DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, NB_CLASS, NB_CORES, trainArguments):
        fusionType, fusionMethod, fusionConfig, monoviewClassifier, monoviewClassifierConfig = trainArguments
        fusionTypeModule = globals()[fusionType]  # Early/late fusion
        trainFusion = getattr(fusionTypeModule, fusionMethod+"Train")  # linearWeighted for example
        classifier = trainFusion(DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, monoviewClassifier,
                                 monoviewClassifierConfig, fusionConfig)
        return fusionType, fusionMethod, classifier

    def predict_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            predictedLabels = self.classifier.predict_hdf5(DATASET, usedIndices=usedIndices)
        else:
            predictedLabels = []
        return predictedLabels

    def predict_probas_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            predictedLabels = self.classifier.predict_probas_hdf5(DATASET, usedIndices=usedIndices)
        else:
            predictedLabels = []
        return predictedLabels

    def predict(self, DATASET, classifier, NB_CLASS):
        fusionType, fusionMethod, fusionClassifier = classifier
        fusionType = globals()[fusionType]  # Early/late fusion
        predictFusion = getattr(fusionType, fusionMethod+"Predict")  # linearWeighted for example
        predictedLabels = predictFusion(DATASET, fusionClassifier)
        return predictedLabels



