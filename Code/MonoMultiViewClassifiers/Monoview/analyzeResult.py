from datetime import timedelta as hms
import numpy as np

from .. import MonoviewClassifiers
from .. import Metrics


def getDBConfigString(name, feat, classificationIndices, shape, classLabelsNames, KFolds):
    learningRate = float(len(classificationIndices[0])) / len(classificationIndices[0]) + len(classificationIndices[1])
    dbConfigString = "Database configuration : \n"
    dbConfigString += "\t- Database name : " + name + "\n"
    dbConfigString += "\t- View name : " + feat + "\t View shape : " + str(shape) + "\n"
    dbConfigString += "\t- Learning Rate : " + str(learningRate) + "\n"
    dbConfigString += "\t- Labels used : " + ", ".join(classLabelsNames) + "\n"
    dbConfigString += "\t- Number of cross validation folds : " + str(KFolds.n_splits) + "\n\n"
    return dbConfigString


def getClassifierConfigString(CL_type, gridSearch, nbCores, nIter, clKWARGS):
    classifierModule = getattr(MonoviewClassifiers, CL_type)
    classifierConfigString = "Classifier configuration : \n"
    classifierConfigString += "\t- " + classifierModule.getConfig(clKWARGS)[5:] + "\n"
    classifierConfigString += "\t- Executed on " + str(nbCores) + " core(s) \n"
    if gridSearch:
        classifierConfigString += "\t- Got configuration using randomized search with " + str(nIter) + " iterations \n"
    classifierConfigString += "\n\n"
    return classifierConfigString


def getMetricScore(metric, y_train, y_train_pred, y_test, y_test_pred):
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    metricScoreTrain = metricModule.score(y_train, y_train_pred)
    metricScoreTest = metricModule.score(y_test, y_test_pred)
    metricScoreString = "\tFor " + metricModule.getConfig(**metricKWARGS) + " : "
    metricScoreString += "\n\t\t- Score on train : " + str(metricScoreTrain)
    metricScoreString += "\n\t\t- Score on test : " + str(metricScoreTest)
    metricScoreString += "\n"
    return metricScoreString


def execute(name, learningRate, KFolds, nbCores, gridSearch, metrics, nIter, feat, CL_type, clKWARGS, classLabelsNames,
            shape, y_train, y_train_pred, y_test, y_test_pred, time, randomState):
    metricsScores = {}
    metricModule = getattr(Metrics, metrics[0][0])
    trainScore = metricModule.score(y_train, y_train_pred)
    testScore = metricModule.score(y_test, y_test_pred)
    stringAnalysis = "Classification on " + name + " database for " + feat + " with " + CL_type + ", random state is " + str(
        randomState) + ".\n\n"
    stringAnalysis += metrics[0][0] + " on train : " + str(trainScore) + "\n" + metrics[0][0] + " on test : " + str(
        testScore) + "\n\n"
    stringAnalysis += getDBConfigString(name, feat, learningRate, shape, classLabelsNames, KFolds)
    stringAnalysis += getClassifierConfigString(CL_type, gridSearch, nbCores, nIter, clKWARGS)
    for metric in metrics:
        stringAnalysis += getMetricScore(metric, y_train, y_train_pred, y_test, y_test_pred)
        if metric[1] is not None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        metricsScores[metric[0]] = [getattr(Metrics, metric[0]).score(y_train, y_train_pred),
                                    getattr(Metrics, metric[0]).score(y_test, y_test_pred)]
    stringAnalysis += "\n\n Classification took " + str(hms(seconds=int(time)))

    imageAnalysis = {}
    return stringAnalysis, imageAnalysis, metricsScores
