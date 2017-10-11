from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
from Methods import *
import Methods.LateFusion
import Metrics



# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def error(testLabels, computedLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)



def printMetricScore(metricScores, metrics):
    metricScoreString = "\n\n"
    for metric in metrics:
        metricModule = getattr(Metrics, metric[0])
        if metric[1]!=None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        metricScoreString += "\tFor "+metricModule.getConfig(**metricKWARGS)+" : "
        metricScoreString += "\n\t\t- Score on train : "+str(metricScores[metric[0]][0])
        metricScoreString += "\n\t\t- Score on test : "+str(metricScores[metric[0]][1])
        metricScoreString += "\n\n"
    return metricScoreString


def getTotalMetricScores(metric, trainLabels, testLabels, DATASET, validationIndices, learningIndices):
    labels = DATASET.get("Labels").value
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}

    trainScore = metricModule.score(labels[learningIndices], trainLabels, **metricKWARGS)
    testScore = metricModule.score(labels[validationIndices], testLabels, **metricKWARGS)
    return [trainScore, testScore]


def getMetricsScores(metrics, trainLabels, testLabels,
                     DATASET, validationIndices, learningIndices):
    metricsScores = {}
    for metric in metrics:
        metricsScores[metric[0]] = getTotalMetricScores(metric, trainLabels, testLabels,
                                                        DATASET, validationIndices, learningIndices)
    return metricsScores


def execute(classifier, trainLabels,
            testLabels, DATASET,
            classificationKWARGS, classificationIndices,
            LABELS_DICTIONARY, views, nbCores, times,
            name, KFolds,
            hyperParamSearch, nIter, metrics,
            viewsIndices, randomState):

    CLASS_LABELS = DATASET.get("Labels").value

    fusionType = classificationKWARGS["fusionType"]
    fusionMethod = classificationKWARGS["fusionMethod"]
    monoviewClassifiersNames = classificationKWARGS["classifiersNames"]
    monoviewClassifiersConfigs = classificationKWARGS["classifiersConfigs"]
    fusionMethodConfig = classificationKWARGS["fusionMethodConfig"]

    # DATASET_LENGTH = DATASET.get("Metadata").attrs["datasetLength"]
    # NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    # kFoldAccuracyOnTrain = np.zeros((nbFolds, statsIter))
    # kFoldAccuracyOnTest = np.zeros((nbFolds, statsIter))
    # kFoldAccuracyOnValidation = np.zeros((nbFolds, statsIter))
    # for statsIterIndex in range(statsIter):
    #
    #         trainIndices = [index for index in range(DATASET_LENGTH) if (index not in fold) and (index not in validationIndices[statsIterIndex])]
    #         trainLabels = CLASS_LABELS[trainIndices]
    #         validationLabels = CLASS_LABELS[ivalidationIndices[statsIterIndex]]
    #         kFoldAccuracyOnTrain[statsIterIndex] = (100 * accuracy_score(trainLabels, trainLabelsIterations[statsIterIndex]))
    #         kFoldAccuracyOnValidation[statsIterIndex] = (100 * accuracy_score(validationLabels,
    #                                                               testLabelsIterations[statsIterIndex]))
    #
    # fusionClassifier = kFoldClassifier[0]
    # fusionConfiguration = fusionClassifier[0].classifier.getConfig(fusionMethodConfig,
    #                                                             monoviewClassifiersNames, monoviewClassifiersConfigs)

    # totalAccuracyOnTrain = np.mean(kFoldAccuracyOnTrain)
    # totalAccuracyOnTest = np.mean(kFoldAccuracyOnTest)
    # totalAccuracyOnValidation = np.mean(kFoldAccuracyOnValidation)
    # extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime = times
    #
    # kFoldLearningTime = [np.mean([kFoldLearningTime[statsIterIndex][foldIdx] for foldIdx in range(nbFolds)])for statsIterIndex in range(statsIter)]
    # kFoldPredictionTime = [np.mean([kFoldPredictionTime[statsIterIndex][foldIdx] for foldIdx in range(nbFolds)])for statsIterIndex in range(statsIter)]
    learningIndices, validationIndices = classificationIndices
    metricModule = getattr(Metrics, metrics[0][0])
    if metrics[0][1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metrics[0][1]))
    else:
        metricKWARGS = {}
    scoreOnTrain = metricModule.score(CLASS_LABELS[learningIndices], CLASS_LABELS[learningIndices], **metricKWARGS)
    scoreOnTest = metricModule.score(CLASS_LABELS[validationIndices], testLabels, **metricKWARGS)
    fusionConfiguration = classifier.classifier.getConfig(fusionMethodConfig,monoviewClassifiersNames, monoviewClassifiersConfigs)
    stringAnalysis = "\t\tResult for Multiview classification with "+ fusionType + " and random state : "+str(randomState)+ \
                     "\n\n"+metrics[0][0]+" :\n\t-On Train : " + str(scoreOnTrain) + "\n\t-On Test : " + str(scoreOnTest) + \
                     "\n\nDataset info :\n\t-Database name : " + name + "\n\t-Labels : " + \
                     ', '.join(LABELS_DICTIONARY.values()) + "\n\t-Views : " + ', '.join(views) + "\n\t-" + str(KFolds.n_splits) + \
                     " folds\n\nClassification configuration : \n\t-Algorithm used : "+fusionType+" "+fusionConfiguration

    if fusionType=="LateFusion":
        stringAnalysis+=Methods.LateFusion.getScores(classifier)
    metricsScores = getMetricsScores(metrics, trainLabels, testLabels,
                                     DATASET, validationIndices, learningIndices)
    # if fusionMethod=="MajorityVoting":
    #     print CLASS_LABELS[learningIndices]==CLASS_LABELS[learningIndices]
    #     import pdb;pdb.set_trace()
    # stringAnalysis += "\n\nComputation time on " + str(nbCores) + " cores : \n\tDatabase extraction time : " + str(
    stringAnalysis+=printMetricScore(metricsScores, metrics)
    #     hms(seconds=int(extractionTime))) + "\n\t"
    # row_format = "{:>15}" * 3
    # stringAnalysis += row_format.format("", *['Learn', 'Prediction'])
    # for index, (learningTime, predictionTime) in enumerate(zip(kFoldLearningTime, kFoldPredictionTime)):
    #     stringAnalysis += '\n\t'
    #     stringAnalysis += row_format.format("Fold " + str(index + 1), *[str(hms(seconds=int(learningTime))),
    #                                                                     str(hms(seconds=int(predictionTime)))])
    # stringAnalysis += '\n\t'
    # stringAnalysis += row_format.format("Total", *[str(hms(seconds=int(sum(kFoldLearningTime)))),
    #                                                str(hms(seconds=int(sum(kFoldPredictionTime))))])
    # stringAnalysis += "\n\tSo a total classification time of " + str(hms(seconds=int(classificationTime))) + ".\n\n"
    imagesAnalysis = {}
    return stringAnalysis, imagesAnalysis, metricsScores
