from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
import Mumbo
from Classifiers import *


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


def execute(classifier, predictedTrainLabels, predictedTestLabels, trainLabels, testLabels, trainData, testData,
            NB_CLASS, trainArguments, LEARNING_RATE, LABELS_DICTIONARY, features, NB_CORES, times):

    trainingSetLength = len(trainLabels)
    testingSetLength = len(testLabels)
    DATASET_LENGTH = trainingSetLength+testingSetLength
    NB_VIEW = len(trainData)

    extractionTime, learningTime, predictionTime, classificationTime = times
    classifierConfigs, NB_ITER, classifierNames = trainArguments
    bestClassifiers, generalAlphas, bestViews = classifier
    #classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
    weakClassifierConfig = [getattr(globals()[classifierName], 'getConfig')(classifierConfig) for classifierConfig, classifierName in zip(classifierConfigs, classifierNames) ]

    stringAnalysis = "\nMumbo classification using "+', '.join(classifierNames)+ 'as weak classifier with config ('+ ' ,'.join(weakClassifierConfig)+'\n\n '+\
                     "Learning on \n\t- "+" ".join(features)+" as features\n\t- "+" ".join(LABELS_DICTIONARY.values())+\
                     " as labels\n\t- "+str(trainingSetLength)+" training examples, "+str(testingSetLength)+\
                     " testing examples ("+str(LEARNING_RATE)+" rate)\n\n With configuration :\n\t- "+str(NB_ITER)+\
                     ' iterations on Mumbo\n\t- '+str(NB_CORES)+' cores used for computing.\n\n'

    stringAnalysis += "The algorithm took : \n\t- "+str(hms(seconds=extractionTime))+" to extract the database,\n\t- "+\
                      str(hms(seconds=learningTime))+" to learn on "+str(trainingSetLength)+" examples and "+str(NB_VIEW)+\
                      " views,\n\t- "+str(hms(seconds=predictionTime))+" to predict on "+str(DATASET_LENGTH)+" examples\n"+\
                      "So a total classification time of "+str(hms(seconds=classificationTime))+".\n\n"

    stringAnalysis += "Total accuracy \n\t- On train : "+str(100*accuracy_score(trainLabels, predictedTrainLabels))+\
                      "%\n"+classification_report(trainLabels, predictedTrainLabels, target_names=LABELS_DICTIONARY.values())+\
                      "\n\t- On test : "+str(100*accuracy_score(testLabels, predictedTestLabels))+"% \n"+ \
                      classification_report(testLabels, predictedTestLabels, target_names=LABELS_DICTIONARY.values())

    predictedTrainLabelsByIter = classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    predictedTestLabelsByIter = classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)

    stringAnalysis += "\n\n\n Analysis for each Mumbo iteration : \n"

    for iterIndex in range(NB_ITER):
        stringAnalysis+= "\t- Iteration "+str(iterIndex+1)+"\n\t\t Accuracy on train : "+\
                         str(accuracy_score(trainLabels, predictedTrainLabelsByIter[iterIndex]))+'\n\t\t Accuracy on test : '+\
                         str(accuracy_score(testLabels, predictedTestLabelsByIter[iterIndex]))+'\n\t\t Selected View : '+\
                         features[int(bestViews[iterIndex])]+"\n"

    name, image = plotAccuracyByIter(predictedTrainLabelsByIter, predictedTestLabelsByIter, trainLabels, testLabels, NB_ITER)
    imagesAnalysis = {}
    imagesAnalysis[name] = image

    return stringAnalysis, imagesAnalysis
