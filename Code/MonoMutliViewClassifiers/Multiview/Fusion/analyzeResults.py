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
        metricScoreString += "\n\t\t- Score on train : "+str(metricScores[metric[0]][0]) +" with STD : "+str(metricScores[metric[0]][3])
        metricScoreString += "\n\t\t- Score on test : "+str(metricScores[metric[0]][1]) +" with STD : "+str(metricScores[metric[0]][4])
        metricScoreString += "\n\t\t- Score on validation : "+str(metricScores[metric[0]][2]) +" with STD : "+str(metricScores[metric[0]][5])
        metricScoreString += "\n\n"
    return metricScoreString


def getTotalMetricScores(metric, kFoldPredictedTrainLabels, kFoldPredictedTestLabels,
                         kFoldPredictedValidationLabels, DATASET, validationIndices, kFolds, statsIter):
    labels = DATASET.get("Labels").value
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    trainScores = []
    testScores = []
    validationScores = []
    for statsIterIndex in range(statsIter):
        trainScores.append(np.mean(np.array([metricModule.score([label for index, label in enumerate(labels) if (index not in fold) and (index not in validationIndices[statsIterIndex])], predictedLabels, **metricKWARGS) for fold, predictedLabels in zip(kFolds[statsIterIndex], kFoldPredictedTrainLabels[statsIterIndex])])))
        testScores.append(np.mean(np.array([metricModule.score(labels[fold], predictedLabels, **metricKWARGS) for fold, predictedLabels in zip(kFolds[statsIterIndex], kFoldPredictedTestLabels[statsIterIndex])])))
        validationScores.append(np.mean(np.array([metricModule.score(labels[validationIndices[statsIterIndex]], predictedLabels, **metricKWARGS) for predictedLabels in kFoldPredictedValidationLabels[statsIterIndex]])))
    return [np.mean(np.array(trainScores)), np.mean(np.array(testScores)), np.mean(np.array(validationScores)), np.std(np.array(testScores)),np.std(np.array(validationScores)), np.std(np.array(trainScores))]


def getMetricsScores(metrics, kFoldPredictedTrainLabels, kFoldPredictedTestLabels,
                     kFoldPredictedValidationLabels, DATASET, validationIndices, kFolds, statsIter):
    metricsScores = {}
    for metric in metrics:
        metricsScores[metric[0]] = getTotalMetricScores(metric, kFoldPredictedTrainLabels, kFoldPredictedTestLabels,
                                                        kFoldPredictedValidationLabels, DATASET, validationIndices, kFolds, statsIter)
    return metricsScores


def execute(kFoldClassifier, kFoldPredictedTrainLabels,
            kFoldPredictedTestLabels, kFoldPredictedValidationLabels,
            DATASET, classificationKWARGS, learningRate, LABELS_DICTIONARY,
            views, nbCores, times, kFolds, name, nbFolds,
            validationIndices, gridSearch, nIter, metrics, statsIter, viewIndices):

    CLASS_LABELS = DATASET.get("Labels").value

    fusionType = classificationKWARGS["fusionType"]
    fusionMethod = classificationKWARGS["fusionMethod"]
    monoviewClassifiersNames = classificationKWARGS["classifiersNames"]
    monoviewClassifiersConfigs = classificationKWARGS["classifiersConfigs"]
    fusionMethodConfig = classificationKWARGS["fusionMethodConfig"]

    DATASET_LENGTH = DATASET.get("Metadata").attrs["datasetLength"]
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    kFoldAccuracyOnTrain = np.zeros((nbFolds, statsIter))
    kFoldAccuracyOnTest = np.zeros((nbFolds, statsIter))
    kFoldAccuracyOnValidation = np.zeros((nbFolds, statsIter))
    for statsIterIndex in range(statsIter):
        for foldIdx, fold in enumerate(kFolds[statsIterIndex]):
            if fold != range(DATASET_LENGTH):
                trainIndices = [index for index in range(DATASET_LENGTH) if (index not in fold) and (index not in validationIndices[statsIterIndex])]
                testLabels = CLASS_LABELS[fold]
                trainLabels = CLASS_LABELS[trainIndices]
                validationLabels = CLASS_LABELS[validationIndices[statsIterIndex]]
                kFoldAccuracyOnTrain[foldIdx, statsIterIndex] = (100 * accuracy_score(trainLabels, kFoldPredictedTrainLabels[statsIterIndex][foldIdx]))
                kFoldAccuracyOnTest[foldIdx, statsIterIndex] = (100 * accuracy_score(testLabels, kFoldPredictedTestLabels[statsIterIndex][foldIdx]))
                kFoldAccuracyOnValidation[foldIdx, statsIterIndex] = (100 * accuracy_score(validationLabels,
                                                                      kFoldPredictedValidationLabels[statsIterIndex][foldIdx]))

    fusionClassifier = kFoldClassifier[0]
    fusionConfiguration = fusionClassifier[0].classifier.getConfig(fusionMethodConfig,
                                                                monoviewClassifiersNames, monoviewClassifiersConfigs)

    totalAccuracyOnTrain = np.mean(kFoldAccuracyOnTrain)
    totalAccuracyOnTest = np.mean(kFoldAccuracyOnTest)
    totalAccuracyOnValidation = np.mean(kFoldAccuracyOnValidation)
    extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime = times

    kFoldLearningTime = [np.mean([kFoldLearningTime[statsIterIndex][foldIdx] for foldIdx in range(nbFolds)])for statsIterIndex in range(statsIter)]
    kFoldPredictionTime = [np.mean([kFoldPredictionTime[statsIterIndex][foldIdx] for foldIdx in range(nbFolds)])for statsIterIndex in range(statsIter)]

    stringAnalysis = "\t\tResult for Multiview classification with "+ fusionType + \
                     "\n\nAverage accuracy :\n\t-On Train : " + str(totalAccuracyOnTrain) + "\n\t-On Test : " + \
                     str(totalAccuracyOnTest) + "\n\t-On Validation : " + str(totalAccuracyOnValidation) + \
                     "\n\nDataset info :\n\t-Database name : " + name + "\n\t-Labels : " + \
                     ', '.join(LABELS_DICTIONARY.values()) + "\n\t-Views : " + ', '.join(views) + "\n\t-" + str(nbFolds) + \
                     " folds\n\nClassification configuration : \n\t-Algorithm used : "+fusionType+" "+fusionConfiguration

    if fusionType=="LateFusion":
        stringAnalysis+=Methods.LateFusion.getAccuracies(kFoldClassifier)
    metricsScores = getMetricsScores(metrics, kFoldPredictedTrainLabels, kFoldPredictedTestLabels,
                                     kFoldPredictedValidationLabels, DATASET, validationIndices, kFolds, statsIter)
    stringAnalysis+=printMetricScore(metricsScores, metrics)
    stringAnalysis += "\n\nComputation time on " + str(nbCores) + " cores : \n\tDatabase extraction time : " + str(
        hms(seconds=int(extractionTime))) + "\n\t"
    row_format = "{:>15}" * 3
    stringAnalysis += row_format.format("", *['Learn', 'Prediction'])
    for index, (learningTime, predictionTime) in enumerate(zip(kFoldLearningTime, kFoldPredictionTime)):
        stringAnalysis += '\n\t'
        stringAnalysis += row_format.format("Fold " + str(index + 1), *[str(hms(seconds=int(learningTime))),
                                                                        str(hms(seconds=int(predictionTime)))])
    stringAnalysis += '\n\t'
    stringAnalysis += row_format.format("Total", *[str(hms(seconds=int(sum(kFoldLearningTime)))),
                                                   str(hms(seconds=int(sum(kFoldPredictionTime))))])
    stringAnalysis += "\n\tSo a total classification time of " + str(hms(seconds=int(classificationTime))) + ".\n\n"
    imagesAnalysis = {}
    return stringAnalysis, imagesAnalysis, metricsScores
