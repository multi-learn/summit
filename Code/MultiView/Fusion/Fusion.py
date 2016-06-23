from Methods import *


def train(DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, NB_CLASS, NB_CORES, trainArguments):
    fusionType, fusionMethod, fusionConfig, monoviewClassifier, monoviewClassifierConfig = trainArguments
    fusionTypeModule = globals()[fusionType]  # Early/late fusion
    trainFusion = getattr(fusionTypeModule, fusionMethod+"Train")  # linearWeighted for example
    classifier = trainFusion(DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, monoviewClassifier,
                             monoviewClassifierConfig, fusionConfig)
    return fusionType, fusionMethod, classifier


def predict(DATASET, classifier, NB_CLASS):
    fusionType, fusionMethod, fusionClassifier = classifier
    fusionType = globals()[fusionType]  # Early/late fusion
    predictFusion = getattr(fusionType, fusionMethod+"Predict")  # linearWeighted for example
    predictedLabels = predictFusion(DATASET, fusionClassifier)
    return predictedLabels
