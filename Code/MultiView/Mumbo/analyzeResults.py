from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
import Mumbo
from Classifiers import *
import logging

def findMainView(bestViews):
    views = list(set(bestViews))
    mainView = ()
    viewCount = np.array([bestView.count(view) for view in views])
    mainView = views[np.argmax(viewCount)]
    return mainView


def plotAccuracyByIter(trainAccuracy, testAccuracy, NB_ITER, bestViews):
    x = range(NB_ITER)
    mainView = findMainView(bestViews)
    figure = plt.figure()
    ax1 = figure.add_subplot(111)

    ax1.set_title("Accuracy depending on iteration")

    ax1.set_xlabel("Iteration Index")
    ax1.set_ylabel("Accuracy")
    ax1.plot(x, trainAccuracy, c='red', label='Train')
    ax1.plot(x, testAccuracy, c='black', label='Test')
    for label, x, y in zip(bestViews, x, trainAccuracy):
        if label != mainView:
            plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    return 'accuracyByIteration', figure


def classifyMumbobyIter(DATASET, classifiers, alphas, views, NB_CLASS):
    DATASET_LENGTH = len(DATASET[0])
    NB_ITER = len(classifiers)
    predictedLabels = np.zeros((DATASET_LENGTH, NB_ITER))
    votes = np.zeros((DATASET_LENGTH, NB_CLASS))

    for classifier, alpha, view, iterIndice in zip(classifiers, alphas, views, range(NB_ITER)):
        votesByIter = np.zeros((DATASET_LENGTH, NB_CLASS))

        for exampleIndice in range(DATASET_LENGTH):
            data = np.array([np.array(DATASET[int(view)][exampleIndice])])
            votesByIter[exampleIndice, int(classifier.predict(data))] += alpha
            votes[exampleIndice] = votes[exampleIndice] + np.array(votesByIter[exampleIndice])
            predictedLabels[exampleIndice, iterIndice] = np.argmax(votes[exampleIndice])

    return np.transpose(predictedLabels)


def error(testLabels, computedLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)


def execute(kFoldClassifier, kFoldPredictedTrainLabels, kFoldPredictedTestLabels, DATASET, CLASS_LABELS,
            NB_CLASS, trainArguments, LEARNING_RATE, LABELS_DICTIONARY, features, NB_CORES, times, NB_VIEW, kFolds):
    extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime = times
    classifierConfigs, NB_ITER, classifierNames = trainArguments
    if LEARNING_RATE<1.0:
        nbFolds = 2
    elif LEARNING_RATE>1.0:
        nbFolds = int(LEARNING_RATE)
    else:
        nbFolds = 1
    DATASET_LENGTH = len(CLASS_LABELS)
    kFoldPredictedTrainLabelsByIter = []
    kFoldPredictedTestLabelsByIter = []
    kFoldBestClassifiers = []
    kFoldGeneralAlphas = []
    kFoldBestViews = []
    kFoldAccuracyOnTrain = []
    kFoldAccuracyOnTest = []
    kFoldAccuracyOnTrainByIter = []
    kFoldAccuracyOnTestByIter = []
    for foldIdx, fold in enumerate(kFolds):
        if fold:
            bestClassifiers, generalAlphas, bestViews = kFoldClassifier[foldIdx]
            trainIndices = [index for index in range(DATASET_LENGTH) if index not in fold]
            testData = dict((key, value[fold, :]) for key, value in DATASET.iteritems())
            trainData = dict((key, value[trainIndices, :]) for key, value in DATASET.iteritems())
            testLabels = CLASS_LABELS[fold]
            trainLabels = CLASS_LABELS[trainIndices]
            PredictedTrainLabelsByIter = classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews,
                                                                       NB_CLASS)
            kFoldPredictedTrainLabelsByIter.append(PredictedTrainLabelsByIter)
            PredictedTestLabelsByIter = classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews,
                                                                      NB_CLASS)
            kFoldPredictedTestLabelsByIter.append(PredictedTestLabelsByIter)
            kFoldAccuracyOnTrainByIter.append([])
            kFoldAccuracyOnTestByIter.append([])
            for iterIndex in range(NB_ITER):
                kFoldAccuracyOnTestByIter[foldIdx].append(100 * accuracy_score(testLabels, PredictedTestLabelsByIter[iterIndex]))
                kFoldAccuracyOnTrainByIter[foldIdx].append(100 * accuracy_score(trainLabels, PredictedTrainLabelsByIter[iterIndex]))
            kFoldBestClassifiers.append(bestClassifiers)
            kFoldGeneralAlphas.append(generalAlphas)
            kFoldBestViews.append(bestViews)
            kFoldAccuracyOnTrain.append(100 * accuracy_score(trainLabels, kFoldPredictedTrainLabels[foldIdx]))
            kFoldAccuracyOnTest.append(100 * accuracy_score(testLabels, kFoldPredictedTestLabels[foldIdx]))

    totalAccuracyOnTrain = np.mean(kFoldAccuracyOnTrain)
    totalAccuracyOnTest = np.mean(kFoldAccuracyOnTest)
    extractionTime, kFoldLearningTime, kFoldPredictionTime, classificationTime = times
    classifierConfigs, NB_ITER, classifierNames = trainArguments
    weakClassifierConfigs = [getattr(globals()[classifierName], 'getConfig')(classifierConfig) for classifierConfig, classifierName in zip(classifierConfigs, classifierNames) ]
    classifierAnalysis = ["\n\t\t-"+classifierName+" "+weakClassifierConfig for classifierName, weakClassifierConfig in zip(classifierNames, weakClassifierConfigs)]
    bestViews = 
    stringAnalysis = "\t\tResult for Multiview classification with Mumbo" \
                     "\n\nAverage accuracy :\n\t-On Train : "+str(totalAccuracyOnTrain)+"\n\t-On Test : "+str(totalAccuracyOnTest)+ \
                     "\n\nDataset info :\n\t-Database name : "+"\n\t-Labels : "+', '.join(LABELS_DICTIONARY.values())+"\n\t-Views : "+', '.join(features)+"\n\t-"+str(nbFolds)+" folds"\
                     "\n\nClassification configuration : \n\t-Algorithm used : Mumbo \n\t-Iterations : "+str(NB_ITER)+"\n\t-Weak Classifiers : "+" ".join(classifierAnalysis)+ \
            "\n\n For each iteration : "

    for iterIndex in range(NB_ITER):
        stringAnalysis += "\n\t- Iteration " + str(iterIndex + 1)
        for foldIdx in range(nbFolds):
            stringAnalysis+= "\n\t\t Fold "+str(foldIdx+1)+"\n\t\t\tAccuracy on train : "+ \
                             str(kFoldAccuracyOnTrainByIter[foldIdx][iterIndex])+'\n\t\t\tAccuracy on test : '+ \
                             str(kFoldAccuracyOnTestByIter[foldIdx][iterIndex])+'\n\t\t\tSelected View : '+ \
                             features[int(kFoldBestViews[foldIdx][iterIndex])]
        stringAnalysis += "\n\t\t- Mean : \n\t\t\t Accuracy on train : "+str(np.array(kFoldAccuracyOnTrainByIter)[:,iterIndex].mean())+\
                          "\n\t\t\t Accuracy on test : "+str(np.array(kFoldAccuracyOnTestByIter)[:,iterIndex].mean())

    stringAnalysis += "\n\nComputation time on "+str(NB_CORES)+" cores : \n\tDatabase extraction time : "+str(hms(seconds=extractionTime))+"\n\t"
    row_format = "{:>15}" * 3
    stringAnalysis += row_format.format("", *['' ,'Learn', 'Prediction'])
    for index, (learningTime, predictionTime) in enumerate(zip(kFoldLearningTime, kFoldPredictionTime)):
         stringAnalysis+='\n\t'
         stringAnalysis+=row_format.format("Fold "+str(index+1), *[str(hms(seconds=learningTime)), str(hms(seconds=predictionTime))     ])
    stringAnalysis+="\n\tSo a total classification time of "+str(hms(seconds=classificationTime))+".\n\n"


    trainAccuracyByIter = np.array(kFoldAccuracyOnTrainByIter).mean(axis=0)
    testAccuracyByIter = np.array(kFoldAccuracyOnTestByIter).mean(axis=0)
    name, image = plotAccuracyByIter(trainAccuracyByIter, testAccuracyByIter, NB_ITER)
    imagesAnalysis = {name: image}

    return stringAnalysis, imagesAnalysis
