from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
import Mumbo
from Classifiers import *
import logging


def plotAccuracyByIter(predictedTrainLabelsByIter, predictedTestLabelsByIter, trainLabels, testLabels, NB_ITER):
    x = range(NB_ITER)
    trainErrors = []
    testErrors = []
    for iterTrain, iterTest in zip(predictedTrainLabelsByIter, predictedTestLabelsByIter):
        pTr, r, f1, s = precision_recall_fscore_support(trainLabels, iterTrain)
        pTe, r, f1, s = precision_recall_fscore_support(testLabels, iterTest)

        trainErrors.append(np.mean(pTr))
        testErrors.append(np.mean(pTe))

    figure = plt.figure()
    ax1 = figure.add_subplot(111)

    ax1.set_title("Accuracy depending on iteration")

    ax1.set_xlabel("Iteration Index")
    ax1.set_ylabel("Accuracy")
    ax1.plot(x,trainErrors, c='red', label='Train')
    ax1.plot(x,testErrors, c='black', label='Test')

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
    classifierAnalysis = ["\n\t- "+classifierName+" "+weakClassifierConfig for classifierName, weakClassifierConfig in zip(classifierNames, weakClassifierConfigs)]

    stringAnalysis = "\t\tResult for Multiview classification with Mumbo" \
                     "\n\nAverage accuracy :\n\t-On Train : "+str(totalAccuracyOnTrain)+"\n\t-On Test : "+str(totalAccuracyOnTest)+ \
                     "\n\nDataset info :\n\t-Database name : "+"\n\t-Labels : "+', '.join(LABELS_DICTIONARY.values())+"\n\t- Views : "+', '.join(features)+"\n\t-"+str(nbFolds)+" folds"\
                     "\n\nClassification configuration : \n\t- Algorithm used : Mumbo \n\t-Iterations : "+str(NB_ITER)+"\n\t-Weak Classifiers : "+" ".join(classifierAnalysis)+ \
            "\n\n For each iteration : "

    for iterIndex in range(NB_ITER):
        stringAnalysis += "\n\t- Iteration " + str(iterIndex + 1)
        for foldIdx in range(nbFolds):
            stringAnalysis+= "\n\t\t Fold "+str(foldIdx+1)+"\n\t\t\tAccuracy on train : "+ \
                             str(kFoldAccuracyOnTrainByIter[foldIdx][iterIndex])+'\n\t\t\tAccuracy on test : '+ \
                             str(kFoldAccuracyOnTestByIter[foldIdx][iterIndex])+'\n\t\t\t Selected View : '+ \
                             str(kFoldBestViews[foldIdx][iterIndex])
        stringAnalysis += "\n\t\t- Mean : \n\t\t\t Accuracy on train : "+str(np.mean(np.array(kFoldAccuracyOnTrain[:][iterIndex])))+\
                          "\n\t\t\t Accuracy on test : "+str(np.mean(np.array(kFoldAccuracyOnTest[:][iterIndex])))

    stringAnalysis += "\n\nComputation time on "+str(NB_CORES)+" cores : \n\tDatabase extraction time : "+str(hms(seconds=extractionTime))+"\n\t"
    row_format = "{:>15}" * 3
    stringAnalysis += row_format.format("", *['' ,'Learn', 'Prediction'])
    for index, (learningTime, predictionTime) in enumerate(zip(kFoldLearningTime, kFoldPredictionTime)):
         stringAnalysis+='\n\t'
         stringAnalysis+=row_format.format("Fold "+str(index+1), *[str(hms(seconds=learningTime)), str(hms(seconds=predictionTime))     ])
    stringAnalysis+="\n\tSo a total classification time of "+str(hms(seconds=classificationTime))+".\n\n"

    logging.info(stringAnalysis)

    #
    # stringAnalysis += "Total accuracy \n\t- On train : "+str(100*accuracy_score(trainLabels, predictedTrainLabels))+ \
    #                   "%\n"+classification_report(trainLabels, predictedTrainLabels, target_names=LABELS_DICTIONARY.values())+ \
    #                   "\n\t- On test : "+str(100*accuracy_score(testLabels, predictedTestLabels))+"% \n"+ \
    #                   classification_report(testLabels, predictedTestLabels, target_names=LABELS_DICTIONARY.values())
    #
    #
    # for iterIndex in range(NB_ITER):
    #     stringAnalysis+= "\t- Iteration "+str(iterIndex+1)+"\n\t\t Accuracy on train : "+ \
    #                      str(accuracy_score(trainLabels, predictedTrainLabelsByIter[iterIndex]))+'\n\t\t Accuracy on test : '+ \
    #                      str(accuracy_score(testLabels, predictedTestLabelsByIter[iterIndex]))+'\n\t\t Selected View : '+ \
    #                      features[int(bestViews[iterIndex])]+"\n"

    name, image = plotAccuracyByIter(predictedTrainLabelsByIter, predictedTestLabelsByIter, trainLabels, testLabels, NB_ITER)
    imagesAnalysis = {}
    imagesAnalysis[name] = image

    return stringAnalysis, imagesAnalysis
