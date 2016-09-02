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

def getMetricScore(metric, y_train, y_train_pred, y_test, y_test_pred):
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    metricScoreString = "\tFor "+metricModule.getConfig(**metricKWARGS)+" : "
    metricScoreString += "\n\t\t- Score on train : "+str(metricModule.score(y_train, y_train_pred))
    metricScoreString += "\n\t\t- Score on test : "+str(metricModule.score(y_test, y_test_pred))
    metricScoreString += "\n"
    return metricScoreString


def execute(kFoldClassifier, kFoldPredictedTrainLabels,
            kFoldPredictedTestLabels, kFoldPredictedValidationLabels,
            DATASET, classificationKWARGS, learningRate, LABELS_DICTIONARY,
            views, nbCores, times, kFolds, name, nbFolds,
            validationIndices, gridSearch, nIter, metrics):

    CLASS_LABELS = DATASET.get("labels").value

    fusionType = classificationKWARGS["fusionType"]
    fusionMethod = classificationKWARGS["fusionMethod"]
    monoviewClassifiersNames = classificationKWARGS["classifiersNames"]
    monoviewClassifiersConfigs = classificationKWARGS["classifiersConfigs"]
    fusionMethodConfig = classificationKWARGS["fusionMethodConfig"]

    DATASET_LENGTH = DATASET.get("Metadata").attrs["datasetLength"]-len(validationIndices)
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    kFoldAccuracyOnTrain = []
    kFoldAccuracyOnTest = []
    kFoldAccuracyOnValidation = []
    for foldIdx, fold in enumerate(kFolds):
        if fold != range(DATASET_LENGTH):
            trainIndices = [index for index in range(DATASET_LENGTH) if index not in fold]
            testLabels = CLASS_LABELS[fold]
            trainLabels = CLASS_LABELS[trainIndices]
            validationLabels = CLASS_LABELS[validationIndices]
            kFoldAccuracyOnTrain.append(100 * accuracy_score(trainLabels, kFoldPredictedTrainLabels[foldIdx]))
            kFoldAccuracyOnTest.append(100 * accuracy_score(testLabels, kFoldPredictedTestLabels[foldIdx]))
            kFoldAccuracyOnValidation.append(100 * accuracy_score(validationLabels,
                                                                  kFoldPredictedValidationLabels[foldIdx]))

    fusionClassifier = kFoldClassifier[0]
    fusionConfiguration = fusionClassifier.classifier.getConfig(fusionMethodConfig,
                                                                monoviewClassifiersNames, monoviewClassifiersConfigs)

    totalAccuracyOnTrain = np.mean(kFoldAccuracyOnTrain)
    totalAccuracyOnTest = np.mean(kFoldAccuracyOnTest)
    totalAccuracyOnValidation = np.mean(kFoldAccuracyOnValidation)
    extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime = times

    stringAnalysis = "\t\tResult for Multiview classification with "+ fusionType + \
                     "\n\nAverage accuracy :\n\t-On Train : " + str(totalAccuracyOnTrain) + "\n\t-On Test : " + \
                     str(totalAccuracyOnTest) + "\n\t-On Validation : " + str(totalAccuracyOnValidation) + \
                     "\n\nDataset info :\n\t-Database name : " + name + "\n\t-Labels : " + \
                     ', '.join(LABELS_DICTIONARY.values()) + "\n\t-Views : " + ', '.join(views) + "\n\t-" + str(nbFolds) + \
                     " folds\n\nClassification configuration : \n\t-Algorithm used : "+fusionType+" "+fusionConfiguration

    if fusionType=="LateFusion":
        stringAnalysis+=Methods.LateFusion.getAccuracies(kFoldClassifier)

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
    return stringAnalysis, imagesAnalysis, totalAccuracyOnTrain, totalAccuracyOnTest, totalAccuracyOnValidation
