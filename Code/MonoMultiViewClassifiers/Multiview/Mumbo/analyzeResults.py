import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
# import logging

# import Mumbo
from . import Classifiers
from ... import Metrics
from ...utils.Dataset import getV, getShape

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def findMainView(bestViews):
    views = list(set(bestViews))
    viewCount = np.array([list(bestViews).count(view) for view in views])
    mainView = views[np.argmax(viewCount)]
    return mainView


def plotAccuracyByIter(scoresOnTainByIter, scoresOnTestByIter, features, classifierAnalysis):
    x = range(len(scoresOnTainByIter))
    figure = plt.figure()
    ax1 = figure.add_subplot(111)
    axes = figure.gca()
    axes.set_ylim([0.40, 1.00])
    titleString = ""
    for view, classifierConfig in zip(features, classifierAnalysis):
        titleString += "\n" + view + " : " + classifierConfig

    ax1.set_title("Score depending on iteration", fontsize=20)
    plt.text(0.5, 1.08, titleString,
             horizontalalignment='center',
             fontsize=8,
             transform=ax1.transAxes)
    figure.subplots_adjust(top=0.8)
    ax1.set_xlabel("Iteration Index")
    ax1.set_ylabel("Accuracy")
    ax1.plot(x, scoresOnTainByIter, c='red', label='Train')
    ax1.plot(x, scoresOnTestByIter, c='black', label='Test')

    ax1.legend(loc='lower center',
               ncol=3, fancybox=True, shadow=True)

    return '-accuracyByIteration', figure


def classifyMumbobyIter_hdf5(usedIndices, DATASET, classifiers, alphas, views, NB_CLASS):
    DATASET_LENGTH = len(usedIndices)
    NB_ITER = len(classifiers)
    predictedLabels = np.zeros((DATASET_LENGTH, NB_ITER))
    votes = np.zeros((DATASET_LENGTH, NB_CLASS))

    for classifier, alpha, view, iterIndex in zip(classifiers, alphas, views, range(NB_ITER)):
        votesByIter = np.zeros((DATASET_LENGTH, NB_CLASS))

        for usedExampleIndex, exampleIndex in enumerate(usedIndices):
            data = np.array([np.array(getV(DATASET, int(view), exampleIndex))])
            votesByIter[usedExampleIndex, int(classifier.predict(data))] += alpha
            votes[usedExampleIndex] = votes[usedExampleIndex] + np.array(votesByIter[usedExampleIndex])
            predictedLabels[usedExampleIndex, iterIndex] = np.argmax(votes[usedExampleIndex])

    return np.transpose(predictedLabels)


def error(testLabels, computedLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)


def getDBConfig(DATASET, LEARNING_RATE, nbFolds, databaseName, validationIndices, LABELS_DICTIONARY):
    nbView = DATASET.get("Metadata").attrs["nbView"]
    viewNames = [DATASET.get("View" + str(viewIndex)).attrs["name"] for viewIndex in range(nbView)]
    viewShapes = [getShape(DATASET, viewIndex) for viewIndex in range(nbView)]
    DBString = "Dataset info :\n\t-Dataset name : " + databaseName
    DBString += "\n\t-Labels : " + ', '.join(LABELS_DICTIONARY.values())
    DBString += "\n\t-Views : " + ', '.join([viewName + " of shape " + str(viewShape)
                                             for viewName, viewShape in zip(viewNames, viewShapes)])
    DBString += "\n\t-" + str(nbFolds) + " folds"
    DBString += "\n\t- Validation set length : " + str(len(validationIndices)) + " for learning rate : " + str(
        LEARNING_RATE) + " on a total number of examples of " + str(DATASET.get("Metadata").attrs["datasetLength"])
    DBString += "\n\n"
    return DBString, viewNames


def getAlgoConfig(classifier, classificationKWARGS, nbCores, viewNames, hyperParamSearch, nIter, times):
    maxIter = classificationKWARGS["maxIter"]
    minIter = classificationKWARGS["minIter"]
    threshold = classificationKWARGS["threshold"]
    extractionTime, classificationTime = times
    weakClassifierConfigs = [getattr(getattr(Classifiers, classifierName), 'getConfig')(classifiersConfig) for classifiersConfig,
                                                                                                    classifierName
                             in zip(classifier.classifiersConfigs, classifier.classifiersNames)]
    classifierAnalysis = [classifierName + " " + weakClassifierConfig + "on " + feature for classifierName,
                                                                                            weakClassifierConfig,
                                                                                            feature
                          in zip(classifier.classifiersNames, weakClassifierConfigs, viewNames)]
    gridSearchString = ""
    if hyperParamSearch:
        gridSearchString += "Configurations found by randomized search with " + str(nIter) + " iterations"
    algoString = "\n\nMumbo configuration : \n\t-Used " + str(nbCores) + " core(s)"
    algoString += "\n\t-Iterations : min " + str(minIter) + ", max " + str(maxIter) + ", threshold " + str(threshold)
    algoString += "\n\t-Weak Classifiers : " + "\n\t\t-".join(classifierAnalysis)
    algoString += "\n\n"
    algoString += "\n\nComputation time on " + str(nbCores) + " cores : \n\tDatabase extraction time : " + str(
        hms(seconds=int(extractionTime))) + "\n\t"
    row_format = "{:>15}" * 3
    algoString += row_format.format("", *['Learn', 'Prediction'])
    algoString += '\n\t'
    algoString += "\n\tSo a total classification time of " + str(hms(seconds=int(classificationTime))) + ".\n\n"
    algoString += "\n\n"
    return algoString, classifierAnalysis


def getReport(classifier, CLASS_LABELS, classificationIndices, DATASET, trainLabels,
              testLabels, viewIndices, metric):
    learningIndices, validationIndices = classificationIndices
    nbView = len(viewIndices)
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    metricModule = getattr(Metrics, metric[0])
    fakeViewsIndicesDict = dict(
        (viewIndex, fakeViewIndex) for viewIndex, fakeViewIndex in zip(viewIndices, range(nbView)))
    trainScore = metricModule.score(CLASS_LABELS[learningIndices], trainLabels)
    testScore = metricModule.score(CLASS_LABELS[validationIndices], testLabels)
    mumboClassifier = classifier
    maxIter = mumboClassifier.iterIndex
    meanAverageAccuracies = np.mean(mumboClassifier.averageScores, axis=0)
    viewsStats = np.array([float(list(mumboClassifier.bestViews).count(viewIndex)) /
                           len(mumboClassifier.bestViews) for viewIndex in range(nbView)])
    PredictedTrainLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET, fakeViewsIndicesDict,
                                                                          usedIndices=learningIndices,
                                                                          NB_CLASS=NB_CLASS)
    PredictedTestLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET, fakeViewsIndicesDict,
                                                                         usedIndices=validationIndices,
                                                                         NB_CLASS=NB_CLASS)
    scoresByIter = np.zeros((len(PredictedTestLabelsByIter), 2))
    for iterIndex, (iterPredictedTrainLabels, iterPredictedTestLabels) in enumerate(
            zip(PredictedTrainLabelsByIter, PredictedTestLabelsByIter)):
        scoresByIter[iterIndex, 0] = metricModule.score(CLASS_LABELS[learningIndices], iterPredictedTrainLabels)
        scoresByIter[iterIndex, 1] = metricModule.score(CLASS_LABELS[validationIndices], iterPredictedTestLabels)

    scoresOnTainByIter = [scoresByIter[iterIndex, 0] for iterIndex in range(maxIter)]

    scoresOnTestByIter = [scoresByIter[iterIndex, 1] for iterIndex in range(maxIter)]

    return (trainScore, testScore, meanAverageAccuracies, viewsStats, scoresOnTainByIter,
            scoresOnTestByIter)


def iterRelevant(iterIndex, kFoldClassifierStats):
    relevants = np.zeros(len(kFoldClassifierStats[0]), dtype=bool)
    for statsIterIndex, kFoldClassifier in enumerate(kFoldClassifierStats):
        for classifierIndex, classifier in enumerate(kFoldClassifier):
            if classifier.iterIndex >= iterIndex:
                relevants[classifierIndex] = True
    return relevants


def modifiedMean(surplusAccuracies):
    maxLen = 0
    for foldAccuracies in surplusAccuracies.values():
        if len(foldAccuracies) > maxLen:
            maxLen = len(foldAccuracies)
    meanAccuracies = []
    for accuracyIndex in range(maxLen):
        accuraciesToMean = []
        for foldIndex in surplusAccuracies.keys():
            try:
                accuraciesToMean.append(surplusAccuracies[foldIndex][accuracyIndex])
            except:
                pass
        meanAccuracies.append(np.mean(np.array(accuraciesToMean)))
    return meanAccuracies


def printMetricScore(metricScores, metrics):
    metricScoreString = "\n\n"
    for metric in metrics:
        metricModule = getattr(Metrics, metric[0])
        if metric[1] is not None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        metricScoreString += "\tFor " + metricModule.getConfig(**metricKWARGS) + " : "
        metricScoreString += "\n\t\t- Score on train : " + str(metricScores[metric[0]][0])
        metricScoreString += "\n\t\t- Score on test : " + str(metricScores[metric[0]][1])
        metricScoreString += "\n\n"
    return metricScoreString


def getTotalMetricScores(metric, trainLabels, testLabels,
                         DATASET, validationIndices, learningIndices):
    labels = DATASET.get("Labels").value
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    validationIndices = validationIndices
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


def getMeanIterations(kFoldClassifierStats, foldIndex):
    iterations = np.array([kFoldClassifier[foldIndex].iterIndex + 1 for kFoldClassifier in kFoldClassifierStats])
    return np.mean(iterations)


def execute(classifier, trainLabels,
            testLabels, DATASET,
            classificationKWARGS, classificationIndices,
            LABELS_DICTIONARY, views, nbCores, times,
            databaseName, KFolds,
            hyperParamSearch, nIter, metrics,
            viewsIndices, randomState):
    learningIndices, validationIndices = classificationIndices
    if classifier.classifiersConfigs is None:
        metricsScores = getMetricsScores(metrics, trainLabels, testLabels,
                                         DATASET, validationIndices, learningIndices)
        return "No good setting for monoview classifier", None, metricsScores
    else:
        LEARNING_RATE = len(learningIndices) / (len(learningIndices) + len(validationIndices))
        nbFolds = KFolds.n_splits

        CLASS_LABELS = DATASET.get("Labels")[...]

        dbConfigurationString, viewNames = getDBConfig(DATASET, LEARNING_RATE, nbFolds, databaseName, validationIndices,
                                                       LABELS_DICTIONARY)
        algoConfigurationString, classifierAnalysis = getAlgoConfig(classifier, classificationKWARGS, nbCores, viewNames,
                                                                    hyperParamSearch, nIter, times)

        (totalScoreOnTrain, totalScoreOnTest, meanAverageAccuracies, viewsStats, scoresOnTainByIter,
         scoresOnTestByIter) = getReport(classifier, CLASS_LABELS, classificationIndices, DATASET,
                                         trainLabels, testLabels, viewsIndices, metrics[0])

        stringAnalysis = "\t\tResult for Multiview classification with Mumbo with random state : " + str(randomState) + \
                         "\n\nAverage " + metrics[0][0] + " :\n\t-On Train : " + str(
            totalScoreOnTrain) + "\n\t-On Test : " + \
                         str(totalScoreOnTest)
        stringAnalysis += dbConfigurationString
        stringAnalysis += algoConfigurationString
        metricsScores = getMetricsScores(metrics, trainLabels, testLabels,
                                         DATASET, validationIndices, learningIndices)
        stringAnalysis += printMetricScore(metricsScores, metrics)
        stringAnalysis += "Mean average scores and stats :"
        for viewIndex, (meanAverageAccuracy, bestViewStat) in enumerate(zip(meanAverageAccuracies, viewsStats)):
            stringAnalysis += "\n\t- On " + viewNames[viewIndex] + \
                              " : \n\t\t- Mean average Accuracy : " + str(meanAverageAccuracy) + \
                              "\n\t\t- Percentage of time chosen : " + str(bestViewStat)
        stringAnalysis += "\n\n For each iteration : "
        for iterIndex in range(len(scoresOnTainByIter)):
            stringAnalysis += "\n\t- Iteration " + str(iterIndex + 1)
            stringAnalysis += "\n\t\tScore on train : " + \
                              str(scoresOnTainByIter[iterIndex]) + '\n\t\tScore on test : ' + \
                              str(scoresOnTestByIter[iterIndex])

        name, image = plotAccuracyByIter(scoresOnTainByIter, scoresOnTestByIter, views, classifierAnalysis)
        imagesAnalysis = {name: image}
        return stringAnalysis, imagesAnalysis, metricsScores
