from .Methods import LateFusion
from ... import Metrics
from ...utils.MultiviewResultAnalysis import printMetricScore, getMetricsScores

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def execute(classifier, trainLabels,
            testLabels, DATASET,
            classificationKWARGS, classificationIndices,
            LABELS_DICTIONARY, views, nbCores, times,
            name, KFolds,
            hyperParamSearch, nIter, metrics,
            viewsIndices, randomState, labels):
    CLASS_LABELS = labels

    fusionType = classificationKWARGS["fusionType"]
    monoviewClassifiersNames = classificationKWARGS["classifiersNames"]
    monoviewClassifiersConfigs = classificationKWARGS["classifiersConfigs"]
    fusionMethodConfig = classificationKWARGS["fusionMethodConfig"]

    learningIndices, validationIndices, testIndicesMulticlass = classificationIndices
    metricModule = getattr(Metrics, metrics[0][0])
    if metrics[0][1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metrics[0][1]))
    else:
        metricKWARGS = {}
    scoreOnTrain = metricModule.score(CLASS_LABELS[learningIndices], CLASS_LABELS[learningIndices], **metricKWARGS)
    scoreOnTest = metricModule.score(CLASS_LABELS[validationIndices], testLabels, **metricKWARGS)
    fusionConfiguration = classifier.classifier.getConfig(fusionMethodConfig, monoviewClassifiersNames,
                                                          monoviewClassifiersConfigs)
    stringAnalysis = "\t\tResult for Multiview classification with " + fusionType + " and random state : " + str(
        randomState) + \
                     "\n\n" + metrics[0][0] + " :\n\t-On Train : " + str(scoreOnTrain) + "\n\t-On Test : " + str(
        scoreOnTest) + \
                     "\n\nDataset info :\n\t-Database name : " + name + "\n\t-Labels : " + \
                     ', '.join(LABELS_DICTIONARY.values()) + "\n\t-Views : " + ', '.join(views) + "\n\t-" + str(
        KFolds.n_splits) + \
                     " folds\n\nClassification configuration : \n\t-Algorithm used : " + fusionType + " " + fusionConfiguration

    if fusionType == "LateFusion":
        stringAnalysis += LateFusion.getScores(classifier)
    metricsScores = getMetricsScores(metrics, trainLabels, testLabels,
                                     validationIndices, learningIndices, labels)
    stringAnalysis += printMetricScore(metricsScores, metrics)

    imagesAnalysis = {}
    return stringAnalysis, imagesAnalysis, metricsScores
