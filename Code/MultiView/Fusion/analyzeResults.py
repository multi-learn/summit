from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
from Methods import *

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

    fusionType, fusionMethod, fusionMethodConfig, monoviewClassifier, fusionClassifierConfig = trainArguments
    bestClassifiers, generalAlphas, bestViews = classifier
    fusionTypeModule = globals()[fusionType]  # Permet d'appeler une fonction avec une string
    monoviewClassifierConfig = getattr(fusionTypeModule, 'getConfig')(fusionMethodConfig, monoviewClassifier,
                                                                  fusionClassifierConfig)
    #monoviewClassifierConfig+'\n\n '+ \
    stringAnalysis = "\n"+fusionType+" classification using "+monoviewClassifier+ 'as monoview classifier '+ \
                     "Learning on \n\t- "+", ".join(features)+" as features\n\t- "+", ".join(LABELS_DICTIONARY.values())+ \
                     " as labels\n\t- "+str(trainingSetLength)+" training examples, "+str(testingSetLength)+ \
                     " testing examples ("+str(LEARNING_RATE)+" rate)\n\n With "+str(NB_CORES)+' cores used for computing.\n\n'

    stringAnalysis += "The algorithm took : \n\t- "+str(hms(seconds=extractionTime))+" to extract the database,\n\t- "+ \
                      str(hms(seconds=learningTime))+" to learn on "+str(trainingSetLength)+" examples and "+str(NB_VIEW)+ \
                      " views,\n\t- "+str(hms(seconds=predictionTime))+" to predict on "+str(DATASET_LENGTH)+" examples\n"+ \
                      "So a total classification time of "+str(hms(seconds=classificationTime))+".\n\n"

    stringAnalysis += "Total accuracy \n\t- On train : "+str(100*accuracy_score(trainLabels, predictedTrainLabels))+ \
                      "%\n"+classification_report(trainLabels, predictedTrainLabels, target_names=LABELS_DICTIONARY.values())+ \
                      "\n\t- On test : "+str(100*accuracy_score(testLabels, predictedTestLabels))+"% \n"+ \
                      classification_report(testLabels, predictedTestLabels, target_names=LABELS_DICTIONARY.values())

    # predictedTrainLabelsByIter = classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    # predictedTestLabelsByIter = classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    #
    # stringAnalysis += "\n\n\n Analysis for each Mumbo iteration : \n"
    #
    # for iterIndex in range(NB_ITER):
    #     stringAnalysis+= "\t- Iteration "+str(iterIndex+1)+"\n\t\t Accuracy on train : "+ \
    #                      str(accuracy_score(trainLabels, predictedTrainLabelsByIter[iterIndex]))+'\n\t\t Accuracy on test : '+ \
    #                      str(accuracy_score(testLabels, predictedTestLabelsByIter[iterIndex]))+'\n\t\t Selected View : '+ \
    #                      features[int(bestViews[iterIndex])]+"\n"
    #
    # name, image = plotAccuracyByIter(predictedTrainLabelsByIter, predictedTestLabelsByIter, trainLabels, testLabels, NB_ITER)
    imagesAnalysis = {}
    # imagesAnalysis[name] = image

    return stringAnalysis, imagesAnalysis
