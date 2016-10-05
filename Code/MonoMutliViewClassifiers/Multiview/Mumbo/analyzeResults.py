from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
import Mumbo
from Classifiers import *
import logging
import Metrics
from utils.Dataset import getV, getShape


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype



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
    axes.set_ylim([40,100])
    titleString = ""
    for view, classifierConfig in zip(features, classifierAnalysis):
        titleString += "\n" + view + " : " + classifierConfig

    ax1.set_title("Score depending on iteration", fontsize=20)
    plt.text(0.5, 1.08, titleString,
             horizontalalignment='center',
             fontsize=8,
             transform = ax1.transAxes)
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
            data = np.array([np.array(getV(DATASET,int(view), exampleIndex))])
            votesByIter[usedExampleIndex, int(classifier.predict(data))] += alpha
            votes[usedExampleIndex] = votes[usedExampleIndex] + np.array(votesByIter[usedExampleIndex])
            predictedLabels[usedExampleIndex, iterIndex] = np.argmax(votes[usedExampleIndex])

    return np.transpose(predictedLabels)


def error(testLabels, computedLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)


def getDBConfig(DATASET, LEARNING_RATE, nbFolds, databaseName, validationIndices, LABELS_DICTIONARY):
    nbView = DATASET.get("Metadata").attrs["nbView"]
    viewNames = [DATASET.get("View"+str(viewIndex)).attrs["name"] for viewIndex in range(nbView)]
    viewShapes = [getShape(DATASET,viewIndex) for viewIndex in range(nbView)]
    DBString = "Dataset info :\n\t-Dataset name : " + databaseName
    DBString += "\n\t-Labels : " + ', '.join(LABELS_DICTIONARY.values())
    DBString += "\n\t-Views : " + ', '.join([viewName+" of shape "+str(viewShape)
                                             for viewName, viewShape in zip(viewNames, viewShapes)])
    DBString += "\n\t-" + str(nbFolds) + " folds"
    DBString += "\n\t- Validation set length : "+str(len(validationIndices[0]))+" for learning rate : "+str(LEARNING_RATE)+" on a total number of examples of "+str(DATASET.get("Metadata").attrs["datasetLength"])
    DBString += "\n\n"
    return DBString, viewNames


def getAlgoConfig(classifiersIterations, initKWARGS, NB_CORES, viewNames, gridSearch, nIter, times):
    classifierNames = initKWARGS["classifiersNames"]
    maxIter = initKWARGS["maxIter"]
    minIter = initKWARGS["minIter"]
    threshold = initKWARGS["threshold"]
    classifiersConfig = initKWARGS["classifiersConfigs"]
    extractionTime, classificationTime = times
    # kFoldLearningTime = [np.mean(np.array([kFoldLearningTime[statsIterIndex][foldIdx]
    #                                        for statsIterIndex in range(len(kFoldLearningTime))]))
    #                                       for foldIdx in range(len(kFoldLearningTime[0]))]
    # kFoldPredictionTime = [np.mean(np.array([kFoldPredictionTime[statsIterIndex][foldIdx]
    #                                        for statsIterIndex in range(len(kFoldPredictionTime))]))
    #                                       for foldIdx in range(len(kFoldPredictionTime[0]))]

    weakClassifierConfigs = [getattr(globals()[classifierName], 'getConfig')(classifiersConfig) for classifiersConfig,
                                                                                                   classifierName
                             in zip(classifiersIterations[0].classifiersConfigs, classifiersIterations[0].classifiersNames)]
    classifierAnalysis = [classifierName + " " + weakClassifierConfig + "on " + feature for classifierName,
                                                                                            weakClassifierConfig,
                                                                                            feature
                          in zip(classifiersIterations[0].classifiersNames, weakClassifierConfigs, viewNames)]
    gridSearchString = ""
    if gridSearch:
        gridSearchString += "Configurations found by randomized search with "+str(nIter)+" iterations"
    algoString = "\n\nMumbo configuration : \n\t-Used "+str(NB_CORES)+" core(s)"
    algoString += "\n\t-Iterations : min " + str(minIter)+ ", max "+str(maxIter)+", threshold "+str(threshold)
    algoString += "\n\t-Weak Classifiers : " + "\n\t\t-".join(classifierAnalysis)
    algoString += "\n\n"
    algoString += "\n\nComputation time on " + str(NB_CORES) + " cores : \n\tDatabase extraction time : " + str(
        hms(seconds=int(extractionTime))) + "\n\t"
    row_format = "{:>15}" * 3
    algoString += row_format.format("", *['Learn', 'Prediction'])
    # for index, (learningTime, predictionTime) in enumerate(zip(kFoldLearningTime, kFoldPredictionTime)):
    #     algoString += '\n\t'
    #     algoString += row_format.format("Fold " + str(index + 1), *[str(hms(seconds=int(learningTime))),
    #                                                                     str(hms(seconds=int(predictionTime)))])
    algoString += '\n\t'
    # algoString += row_format.format("Total", *[str(hms(seconds=int(sum(kFoldLearningTime)))),
    #                                                str(hms(seconds=int(sum(kFoldPredictionTime))))])
    algoString += "\n\tSo a total classification time of " + str(hms(seconds=int(classificationTime))) + ".\n\n"
    algoString += "\n\n"
    return algoString, classifierAnalysis


def getReport(classifiersIterations, CLASS_LABELS, iterationValidationIndices, DATASET, trainLabelsIterations,
              testLabelsIterations, statsIter, viewIndices, metric):
    nbView = len(viewIndices)
    viewsDict = dict((viewIndex, index) for index, viewIndex in enumerate(viewIndices))
    DATASET_LENGTH = DATASET.get("Metadata").attrs["datasetLength"]
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    trainScores = []
    testScores = []
    meanAverageAccuraciesIterations =[]
    viewsStatsIteration = np.zeros((statsIter, nbView))
    scoresByIterIteration = []
    maxIter = 0
    for statIterIndex in range(statsIter):
        predictedTrainLabels = trainLabelsIterations[statIterIndex]
        predictedTestLabels = testLabelsIterations[statIterIndex]
        validationIndices = iterationValidationIndices[statIterIndex]
        learningIndices = [index for index in range(DATASET_LENGTH) if index not in validationIndices]
        trainScore = metricModule.score(CLASS_LABELS[learningIndices], predictedTrainLabels)
        testScore = metricModule.score(CLASS_LABELS[validationIndices], predictedTestLabels)
        trainScores.append(trainScore)
        testScores.append(testScore)
        mumboClassifier = classifiersIterations[statIterIndex]
        if mumboClassifier.iterIndex+1 > maxIter:
            maxIter = mumboClassifier.iterIndex
        meanAverageAccuraciesIterations.append(np.mean(mumboClassifier.averageAccuracies, axis=0))
        viewsStatsIteration[statIterIndex, :] = np.array([float(list(mumboClassifier.bestViews).count(viewIndex))/
                                                      len(mumboClassifier.bestViews)for viewIndex in range(nbView)])
        PredictedTrainLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET, usedIndices=learningIndices,
                                                                              NB_CLASS=NB_CLASS)
        PredictedTestLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET, usedIndices=validationIndices,
                                                                              NB_CLASS=NB_CLASS)
        scoresByIter = np.zeros((len(PredictedTestLabelsByIter),2))
        for iterIndex,(iterPredictedTrainLabels, iterPredictedTestLabels) in enumerate(zip(PredictedTrainLabelsByIter, PredictedTestLabelsByIter)):
            scoresByIter[iterIndex, 0] = metricModule.score(CLASS_LABELS[learningIndices],iterPredictedTrainLabels)
            scoresByIter[iterIndex, 1] = metricModule.score(CLASS_LABELS[validationIndices],iterPredictedTestLabels)
        scoresByIterIteration.append(scoresByIter)

    scoresOnTainByIter = [np.mean(np.array([scoresByIterIteration[statsIterIndex][iterIndex, 0]
                                            for statsIterIndex in range(statsIter)
                                            if scoresByIterIteration[statsIterIndex].shape[0]>iterIndex]))
                          for iterIndex in range(maxIter)]

    scoresOnTestByIter = [np.mean(np.array([scoresByIterIteration[statsIterIndex][iterIndex, 1]
                                            for statsIterIndex in range(statsIter)
                                            if scoresByIterIteration[statsIterIndex].shape[0]>iterIndex]))
                          for iterIndex in range(maxIter)]

    viewsStats = np.mean(viewsStatsIteration, axis=0)
    meanAverageAccuracies = np.mean(np.array(meanAverageAccuraciesIterations), axis=0)
    totalScoreOnTrain = np.mean(np.array(trainScores))
    totalScoreOnTest = np.mean(np.array(trainScores))
    return (totalScoreOnTrain, totalScoreOnTest, meanAverageAccuracies, viewsStats, scoresOnTainByIter,
            scoresOnTestByIter)



def getClassificationReport(kFolds, kFoldClassifier, CLASS_LABELS, validationIndices, DATASET,
                            kFoldPredictedTrainLabels, kFoldPredictedTestLabels, kFoldPredictedValidationLabels,statsIter, viewIndices):
    nbView = len(viewIndices)
    viewsDict = dict((viewIndex, index) for index, viewIndex in enumerate(viewIndices))
    DATASET_LENGTH = DATASET.get("Metadata").attrs["datasetLength"]
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    iterKFoldBestViews = []
    iterKFoldMeanAverageAccuracies = []
    iterKFoldAccuracyOnTrainByIter = []
    iterKFoldAccuracyOnTestByIter = []
    iterKFoldAccuracyOnValidationByIter = []
    iterKFoldBestViewsStats = []
    totalAccuracyOnTrainIter = []
    totalAccuracyOnTestIter = []
    totalAccuracyOnValidationIter = []

    for statIterIndex in range(statsIter):
        kFoldPredictedTrainLabelsByIter = []
        kFoldPredictedTestLabelsByIter = []
        kFoldPredictedValidationLabelsByIter = []
        kFoldBestViews = []
        kFoldAccuracyOnTrain = []
        kFoldAccuracyOnTest = []
        kFoldAccuracyOnValidation = []
        kFoldAccuracyOnTrainByIter = []
        kFoldAccuracyOnTestByIter = []
        kFoldAccuracyOnValidationByIter = []
        kFoldMeanAverageAccuracies = []
        kFoldBestViewsStats = []
        for foldIdx, fold in enumerate(kFolds[statIterIndex]):
            if fold != range(DATASET_LENGTH):

                trainIndices = [index for index in range(DATASET_LENGTH) if (index not in fold) and (index not in validationIndices[statIterIndex])]
                testLabels = CLASS_LABELS[fold]
                trainLabels = CLASS_LABELS[trainIndices]
                validationLabels = CLASS_LABELS[validationIndices[statIterIndex]]

                mumboClassifier = kFoldClassifier[statIterIndex][foldIdx]
                kFoldBestViews.append(mumboClassifier.bestViews)
                meanAverageAccuracies = np.mean(mumboClassifier.averageAccuracies, axis=0)
                kFoldMeanAverageAccuracies.append(meanAverageAccuracies)
                kFoldBestViewsStats.append([float(list(mumboClassifier.bestViews).count(viewIndex))/
                                            len(mumboClassifier.bestViews)
                                            for viewIndex in range(nbView)])

                kFoldAccuracyOnTrain.append(100 * accuracy_score(trainLabels, kFoldPredictedTrainLabels[statIterIndex][foldIdx]))
                kFoldAccuracyOnTest.append(100 * accuracy_score(testLabels, kFoldPredictedTestLabels[statIterIndex][foldIdx]))
                kFoldAccuracyOnValidation.append(100 * accuracy_score(validationLabels,
                                                                      kFoldPredictedValidationLabels[statIterIndex][foldIdx]))

                PredictedTrainLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET, usedIndices=trainIndices,
                                                                                      NB_CLASS=NB_CLASS)
                kFoldPredictedTrainLabelsByIter.append(PredictedTrainLabelsByIter)
                PredictedTestLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET, usedIndices=fold,
                                                                                     NB_CLASS=NB_CLASS)
                kFoldPredictedTestLabelsByIter.append(PredictedTestLabelsByIter)
                PredictedValidationLabelsByIter = mumboClassifier.classifyMumbobyIter_hdf5(DATASET,
                                                                                           usedIndices=validationIndices[statIterIndex],
                                                                                           NB_CLASS=NB_CLASS)
                kFoldPredictedValidationLabelsByIter.append(PredictedValidationLabelsByIter)

                kFoldAccuracyOnTrainByIter.append([])
                kFoldAccuracyOnTestByIter.append([])
                kFoldAccuracyOnValidationByIter.append([])
                for iterIndex in range(mumboClassifier.iterIndex+1):
                    if len(PredictedTestLabelsByIter)==mumboClassifier.iterIndex+1:
                        kFoldAccuracyOnTestByIter[foldIdx].append(100 * accuracy_score(testLabels,
                                                                                       PredictedTestLabelsByIter[iterIndex]))
                    else:
                        kFoldAccuracyOnTestByIter[foldIdx].append(0.0)
                    kFoldAccuracyOnTrainByIter[foldIdx].append(100 * accuracy_score(trainLabels,
                                                                                    PredictedTrainLabelsByIter[iterIndex]))
                    kFoldAccuracyOnValidationByIter[foldIdx].append(100 * accuracy_score(validationLabels,
                                                                                         PredictedValidationLabelsByIter[iterIndex]))


        iterKFoldBestViews.append(kFoldBestViews)
        iterKFoldMeanAverageAccuracies.append(kFoldMeanAverageAccuracies)
        iterKFoldAccuracyOnTrainByIter.append(kFoldAccuracyOnTrainByIter)
        iterKFoldAccuracyOnTestByIter.append(kFoldAccuracyOnTestByIter)
        iterKFoldAccuracyOnValidationByIter.append(kFoldAccuracyOnValidationByIter)
        iterKFoldBestViewsStats.append(kFoldBestViewsStats)
        totalAccuracyOnTrainIter.append(np.mean(kFoldAccuracyOnTrain))
        totalAccuracyOnTestIter.append(np.mean(kFoldAccuracyOnTest))
        totalAccuracyOnValidationIter.append(np.mean(kFoldAccuracyOnValidation))
    kFoldMeanAverageAccuraciesM = []
    kFoldBestViewsStatsM = []
    kFoldAccuracyOnTrainByIterM = []
    kFoldAccuracyOnTestByIterM = []
    kFoldAccuracyOnValidationByIterM = []
    kFoldBestViewsM = []
    for foldIdx in range(len(kFolds[0])):
        kFoldBestViewsStatsM.append(np.mean(np.array([iterKFoldBestViewsStats[statIterIndex][foldIdx] for statIterIndex in range(statsIter)]), axis=0))
        bestViewVotes = []
        MeanAverageAccuraciesM = np.zeros((statsIter, nbView))
        AccuracyOnValidationByIterM = []
        AccuracyOnTrainByIterM = []
        AccuracyOnTestByIterM = []
        nbTrainIterations = []
        nbTestIterations = []
        nbValidationIterations = np.zeros(statsIter)
        for statIterIndex in range(statsIter):
            for iterationIndex, viewForIteration in enumerate(iterKFoldBestViews[statIterIndex][foldIdx]):
                if statIterIndex==0:
                    bestViewVotes.append(np.zeros(nbView))
                    bestViewVotes[iterationIndex][viewsDict[viewForIteration]]+=1
                else:
                    bestViewVotes[iterationIndex][viewsDict[viewForIteration]]+=1

            MeanAverageAccuraciesM[statIterIndex] = np.array(iterKFoldMeanAverageAccuracies[statIterIndex][foldIdx])

            for valdiationAccuracyIndex, valdiationAccuracy in enumerate(iterKFoldAccuracyOnValidationByIter[statIterIndex][foldIdx]):
                if statIterIndex==0:
                    AccuracyOnValidationByIterM.append([])
                    AccuracyOnValidationByIterM[valdiationAccuracyIndex].append(valdiationAccuracy)
                else:
                    AccuracyOnValidationByIterM[valdiationAccuracyIndex].append(valdiationAccuracy)
            for trainAccuracyIndex, trainAccuracy in enumerate(iterKFoldAccuracyOnTrainByIter[statIterIndex][foldIdx]):
                if statIterIndex==0:
                    AccuracyOnTrainByIterM.append([])
                    AccuracyOnTrainByIterM[trainAccuracyIndex].append(trainAccuracy)
                else:
                    AccuracyOnTestByIterM[trainAccuracyIndex].append(trainAccuracy)
            for testAccuracyIndex, testAccuracy in enumerate(iterKFoldAccuracyOnTestByIter[statIterIndex][foldIdx]):
                if statIterIndex==0:
                    AccuracyOnTestByIterM.append([])
                    AccuracyOnTestByIterM[testAccuracyIndex].append(testAccuracy)
                else:
                    AccuracyOnTestByIterM[testAccuracyIndex].append(testAccuracy)

            #AccuracyOnValidationByIterM.append(iterKFoldAccuracyOnValidationByIter[statIterIndex][foldIdx])
            #AccuracyOnTrainByIterM.append(iterKFoldAccuracyOnTrainByIter[statIterIndex][foldIdx])
            #AccuracyOnTestByIterM.append(iterKFoldAccuracyOnTestByIter[statIterIndex][foldIdx])

        kFoldAccuracyOnTrainByIterM.append([np.mean(np.array(accuracies)) for accuracies in AccuracyOnTrainByIterM])
        kFoldAccuracyOnTestByIterM.append([np.mean(np.array(accuracies)) for accuracies in AccuracyOnTestByIterM])
        kFoldAccuracyOnValidationByIterM.append([np.mean(np.array(accuracies)) for accuracies in AccuracyOnValidationByIterM])

        kFoldMeanAverageAccuraciesM.append(np.mean(MeanAverageAccuraciesM, axis=0))
        kFoldBestViewsM.append(np.array([np.argmax(bestViewVote) for bestViewVote in bestViewVotes]))


    totalAccuracyOnTrain = np.mean(np.array(totalAccuracyOnTrainIter))
    totalAccuracyOnTest = np.mean(np.array(totalAccuracyOnTestIter))
    totalAccuracyOnValidation = np.mean(np.array(totalAccuracyOnValidationIter))
    return (totalAccuracyOnTrain, totalAccuracyOnTest, totalAccuracyOnValidation, kFoldMeanAverageAccuraciesM,
            kFoldBestViewsStatsM, kFoldAccuracyOnTrainByIterM, kFoldAccuracyOnTestByIterM, kFoldAccuracyOnValidationByIterM,
            kFoldBestViewsM)

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
        if len(foldAccuracies)>maxLen:
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
        if metric[1]!=None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        metricScoreString += "\tFor "+metricModule.getConfig(**metricKWARGS)+" : "
        metricScoreString += "\n\t\t- Score on train : "+str(metricScores[metric[0]][0]) +" with STD : "+str(metricScores[metric[0]][2])
        metricScoreString += "\n\t\t- Score on test : "+str(metricScores[metric[0]][1]) +" with STD : "+str(metricScores[metric[0]][3])
        metricScoreString += "\n\n"
    return metricScoreString


def getTotalMetricScores(metric, trainLabelsIterations, testLabelsIterations, DATASET, iterationValidationIndices, statsIter):
    labels = DATASET.get("Labels").value
    DATASET_LENGTH = DATASET.get("Metadata").attrs["datasetLength"]
    metricModule = getattr(Metrics, metric[0])
    if metric[1]!=None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    trainScores = []
    testScores = []
    validationScores = []
    for statsIterIndex in range(statsIter):
        validationIndices = iterationValidationIndices[statsIterIndex]
        learningIndices = [index for index in range(DATASET_LENGTH) if index not in validationIndices]
        trainScores.append(metricModule.score(labels[learningIndices], trainLabelsIterations[statsIterIndex], **metricKWARGS))
        testScores.append(metricModule.score(labels[validationIndices], testLabelsIterations[statsIterIndex], **metricKWARGS))
    return [np.mean(np.array(trainScores)), np.mean(np.array(testScores)), np.std(np.array(trainScores)),
            np.std(np.array(testScores))]


def getMetricsScores(metrics, trainLabelsIterations, testLabelsIterations,
                     DATASET, validationIndices, statsIter):
    metricsScores = {}
    for metric in metrics:
        metricsScores[metric[0]] = getTotalMetricScores(metric, trainLabelsIterations, testLabelsIterations,
                                                        DATASET, validationIndices, statsIter)
    return metricsScores


def getMeanIterations(kFoldClassifierStats, foldIndex):
    iterations = np.array([kFoldClassifier[foldIndex].iterIndex+1 for kFoldClassifier in kFoldClassifierStats])
    return np.mean(iterations)

def execute(classifiersIterations, trainLabelsIterations,testLabelsIterations, DATASET,initKWARGS,
            LEARNING_RATE,LABELS_DICTIONARY,views, NB_CORES, times,databaseName, nbFolds, validationIndices, gridSearch,
            nIter, metrics, statsIter,viewIndices):

    CLASS_LABELS = DATASET.get("Labels")[...]

    dbConfigurationString, viewNames = getDBConfig(DATASET, LEARNING_RATE, nbFolds, databaseName, validationIndices, LABELS_DICTIONARY)
    algoConfigurationString, classifierAnalysis = getAlgoConfig(classifiersIterations, initKWARGS, NB_CORES, viewNames, gridSearch, nIter, times)


    (totalScoreOnTrain, totalScoreOnTest, meanAverageAccuracies, viewsStats, scoresOnTainByIter,
     scoresOnTestByIter) = getReport(classifiersIterations, CLASS_LABELS, validationIndices, DATASET,
                                                  trainLabelsIterations, testLabelsIterations, statsIter, viewIndices, metrics[0])

    stringAnalysis = "\t\tResult for Multiview classification with Mumbo" \
                     "\n\nAverage "+metrics[0][0]+" :\n\t-On Train : " + str(totalScoreOnTrain) + "\n\t-On Test : " + \
                     str(totalScoreOnTest)
    stringAnalysis += dbConfigurationString
    stringAnalysis += algoConfigurationString
    metricsScores = getMetricsScores(metrics, trainLabelsIterations, testLabelsIterations,
                                     DATASET, validationIndices, statsIter)
    stringAnalysis += printMetricScore(metricsScores, metrics)
    stringAnalysis += "Mean average accuracies and stats :"
    for viewIndex, (meanAverageAccuracy, bestViewStat) in enumerate(zip(meanAverageAccuracies,viewsStats)):
        stringAnalysis+="\n\t- On "+viewNames[viewIndex]+ \
                        " : \n\t\t- Mean average Accuracy : "+str(meanAverageAccuracy)+ \
                        "\n\t\t- Percentage of time chosen : "+str(bestViewStat)
    stringAnalysis += "\n\n For each iteration : "
    for iterIndex in range(len(scoresOnTainByIter)):
        stringAnalysis += "\n\t- Iteration " + str(iterIndex + 1)
        stringAnalysis += "\n\t\tScore on train : " + \
                              str(scoresOnTainByIter[iterIndex]) + '\n\t\tScore on test : ' + \
                              str(scoresOnTestByIter[iterIndex])


    name, image = plotAccuracyByIter(scoresOnTainByIter, scoresOnTestByIter, views, classifierAnalysis)
    imagesAnalysis = {name: image}
    return stringAnalysis, imagesAnalysis, metricsScores
