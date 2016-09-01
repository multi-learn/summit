from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import operator
from datetime import timedelta as hms
from Methods import *

def error(testLabels, computedLabels):
    error = sum(map(operator.ne, computedLabels, testLabels))
    return float(error) * 100 / len(computedLabels)


def execute(kFoldClassifier, kFoldPredictedTrainLabels, kFoldPredictedTestLabels, kFoldPredictedValidationLabels,
            DATASET, initKWARGS, LEARNING_RATE, LABELS_DICTIONARY, views, NB_CORES, times, kFolds, name, nbFolds,
            validationIndices):



    CLASS_LABELS = DATASET["/Labels/labelsArray"][...]
    #NB_ITER, classifierNames, classifierConfigs = initKWARGS.values()
    monoviewClassifiersNames, fusionMethodConfig, fusionMethod, fusionType, monoviewClassifiersConfigs = initKWARGS.values()


    DATASET_LENGTH = DATASET.get("datasetLength").value-len(validationIndices)
    NB_CLASS = DATASET.get("nbClass").value
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




    stringAnalysis += "\n\nComputation time on " + str(NB_CORES) + " cores : \n\tDatabase extraction time : " + str(
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

    # trainingSetLength = len(trainLabels)
    # testingSetLength = len(testLabels)
    # DATASET_LENGTH = trainingSetLength+testingSetLength
    # extractionTime, learningTime, predictionTime, classificationTime = times
    #
    # fusionType, fusionMethod, fusionMethodConfig, monoviewClassifier, fusionClassifierConfig = trainArguments
    # bestClassifiers, generalAlphas, bestViews = classifier
    # fusionTypeModule = globals()[fusionType]  # Permet d'appeler une fonction avec une string
    # monoviewClassifierConfig = getattr(fusionTypeModule, 'getConfig')(fusionMethodConfig, monoviewClassifier,
    #                                                               fusionClassifierConfig)
    # #monoviewClassifierConfig+'\n\n '+ \
    # stringAnalysis = "\n"+fusionType+" classification using "+monoviewClassifier+ 'as monoview classifier '+ \
    #                  "Learning on \n\t- "+", ".join(features)+" as views\n\t- "+", ".join(LABELS_DICTIONARY.values())+ \
    #                  " as labels\n\t- "+str(trainingSetLength)+" training examples, "+str(testingSetLength)+ \
    #                  " testing examples ("+str(LEARNING_RATE)+" rate)\n\n With "+str(NB_CORES)+' cores used for computing.\n\n'
    #
    # stringAnalysis += "The algorithm took : \n\t- "+str(hms(seconds=extractionTime))+" to extract the database,\n\t- "+ \
    #                   str(hms(seconds=learningTime))+" to learn on "+str(trainingSetLength)+" examples and "+str(NB_VIEW)+ \
    #                   " views,\n\t- "+str(hms(seconds=predictionTime))+" to predict on "+str(DATASET_LENGTH)+" examples\n"+ \
    #                   "So a total classification time of "+str(hms(seconds=classificationTime))+".\n\n"
    #
    # stringAnalysis += "Total accuracy \n\t- On train : "+str(100*accuracy_score(trainLabels, predictedTrainLabels))+ \
    #                   "%\n"+classification_report(trainLabels, predictedTrainLabels, target_names=LABELS_DICTIONARY.values())+ \
    #                   "\n\t- On test : "+str(100*accuracy_score(testLabels, predictedTestLabels))+"% \n"+ \
    #                   classification_report(testLabels, predictedTestLabels, target_names=LABELS_DICTIONARY.values())

    # predictedTrainLabelsByIter = classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    # predictedTestLabelsByIter = classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    #
    # stringAnalysis += "\n\n\n Analysis for each Mumbo iteration : \n"
    #
    # for iterIndex in range(NB_ITER):
    #     stringAnalysis+= "\t- Iteration "+str(iterIndex+1)+"\n\t\t Accuracy on train : "+ \
    #                      str(accuracy_score(trainLabels, predictedTrainLabelsByIter[iterIndex]))+'\n\t\t Accuracy on test : '+ \
    #                      str(accuracy_score(testLabels, predictedTestLabelsByIter[iterIndex]))+'\n\t\t Selected View : '+ \
    #                      views[int(bestViews[iterIndex])]+"\n"
    #
    # name, image = plotAccuracyByIter(predictedTrainLabelsByIter, predictedTestLabelsByIter, trainLabels, testLabels, NB_ITER)
    # imagesAnalysis[name] = image

    return stringAnalysis, imagesAnalysis, totalAccuracyOnTrain, totalAccuracyOnTest, totalAccuracyOnValidation
