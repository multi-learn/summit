import numpy as np
import math
from joblib import Parallel, delayed
from Classifiers import *

# Data shape : ((Views, Examples, Corrdinates))


def initialize(NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, CLASS_LABELS):
    costMatrices = np.array([
                    np.array([
                        np.array([
                            np.array([1 if CLASS_LABELS[exampleIndice]!=classe
                                       else -(NB_CLASS-1)
                                       for classe in range(NB_CLASS)
                                    ]) for exampleIndice in range(DATASET_LENGTH)
                            ]) for viewIndice in range(NB_VIEW)]) \
                        if iteration==0 \
                        else np.zeros((NB_VIEW, DATASET_LENGTH, NB_CLASS)) \
                        for iteration in range(NB_ITER+1)
                    ])
    generalCostMatrix = np.array([
                            np.array([
                                np.array([1 if CLASS_LABELS[exampleIndice]!=classe
                                           else -(NB_CLASS-1)
                                           for classe in range(NB_CLASS)
                                ]) for exampleIndice in range(DATASET_LENGTH)
                            ]) for iteration in range(NB_ITER)
                        ])
    fs = np.zeros((NB_ITER, NB_VIEW, DATASET_LENGTH, NB_CLASS))
    ds = np.zeros((NB_ITER, NB_VIEW, DATASET_LENGTH))
    edges = np.zeros((NB_ITER, NB_VIEW))
    alphas = np.zeros((NB_ITER, NB_VIEW))
    predictions = np.zeros((NB_ITER, NB_VIEW, DATASET_LENGTH))
    generalAlphas = np.zeros(NB_ITER)
    generalFs = np.zeros((NB_ITER, DATASET_LENGTH, NB_CLASS))
    return costMatrices, generalCostMatrix, fs, ds, edges, alphas, \
            predictions, generalAlphas, generalFs


def computeWeights(costMatrices, NB_CLASS, DATASET_LENGTH, iterIndice, 
                    viewIndice, CLASS_LABELS):
    dist = np.sum(costMatrices[iterIndice, viewIndice])
    dist = dist - np.sum(np.array([costMatrices[iterIndice, viewIndice, exampleIndice, CLASS_LABELS[exampleIndice]] for exampleIndice in range(DATASET_LENGTH)]))

    weights = np.array([costMatrices[iterIndice, viewIndice,
                                    exampleIndice, CLASS_LABELS[exampleIndice]]/dist\
                         for exampleIndice in range(DATASET_LENGTH)])
    return weights


def trainWeakClassifier(classifierName, monoviewDataset, CLASS_LABELS, costMatrices, 
                        NB_CLASS, DATASET_LENGTH, iterIndice, viewIndice, 
                        classifier_config):
    weights = computeWeights(costMatrices, NB_CLASS, DATASET_LENGTH, 
                            iterIndice, viewIndice, CLASS_LABELS)
    classifierMethod = globals()["DecisionTree"].DecisionTree  # Permet d'appeler une fonction avec une string
    classifier, classes = classifierMethod(monoviewDataset, CLASS_LABELS, classifier_config, weights)
    return classifier, classes



def trainWeakClassifers(classifierName, DATASET, CLASS_LABELS, costMatrices,
                        NB_CLASS, DATASET_LENGTH, iterIndice, classifier_config,
                        NB_CORES, NB_VIEW):
    trainedClassifiers = []
    labelsMatrix = []
    if NB_CORES > NB_VIEW:
        NB_JOBS = NB_VIEW
    else:
        NB_JOBS = NB_CORES

    trainedClassifiersAndLabels = Parallel(n_jobs=NB_JOBS)(
        delayed(trainWeakClassifier)(classifierName, DATASET[viewIndice], CLASS_LABELS, 
                                    costMatrices, NB_CLASS, DATASET_LENGTH, 
                                    iterIndice, viewIndice, classifier_config) 
        for viewIndice in range(NB_VIEW))

    for (classifier, labelsArray) in trainedClassifiersAndLabels:
        trainedClassifiers.append(classifier)
        labelsMatrix.append(labelsArray)

    return np.array(trainedClassifiers), np.array(labelsMatrix)


def computeEdge(predictionMatrix, costMatrix, NB_CLASS, DATASET_LENGTH, CLASS_LABELS):
    # return np.sum(np.array([np.sum(predictionMatrix*costMatrix[:,classIndice]) for classIndice in range(NB_CLASS)]))
    cCost = np.sum(np.array([costMatrix[exampleIndice, predictionMatrix[exampleIndice]] for exampleIndice in range(DATASET_LENGTH)]))
    tCost = np.sum(np.array([-costMatrix[exampleIndice, CLASS_LABELS[exampleIndice]] for exampleIndice in range(DATASET_LENGTH)]))
    if tCost == 0:
        edge = 0.
    else:
        edge = -cCost/tCost
    return edge


def computeAlpha(edge):
    return 0.5*math.log((1+edge)/(1-edge))


def allViewsClassifyWell(predictions, pastIterIndice, NB_VIEW, CLASS_LABEL, 
                        exampleIndice):
    bool = True
    for viewIndice in range(NB_VIEW):
        if predictions[pastIterIndice, viewIndice, exampleIndice]!=CLASS_LABEL:
            bool = False
    return bool


def updateDs(ds, predictions, CLASS_LABELS, NB_VIEW, DATASET_LENGTH, NB_CLASS,
             iterIndice):
    for viewIndice in range(NB_VIEW):
        for exampleIndice in range(DATASET_LENGTH):
            for pastIterIndice in range(iterIndice):
                
                if predictions[pastIterIndice, viewIndice, exampleIndice]\
                 ==\
                   CLASS_LABELS[exampleIndice]\
                 or allViewsClassifyWell(predictions, pastIterIndice, 
                                        NB_VIEW, CLASS_LABELS[exampleIndice],
                                        exampleIndice):

                    ds[pastIterIndice, viewIndice, exampleIndice]=1
                else:
                    ds[pastIterIndice, viewIndice, exampleIndice]=0
    return ds


def updateFs(predictions, ds, alphas, fs, iterIndice, NB_VIEW, DATASET_LENGTH,
             NB_CLASS, CLASS_LABELS):
    for viewIndice in range(NB_VIEW):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                fs[iterIndice, viewIndice, exampleIndice, classe] \
                = np.sum(np.array([alphas[pastIterIndice, viewIndice]\
                                    *ds[pastIterIndice, viewIndice, exampleIndice] \
                                    if predictions[pastIterIndice, viewIndice, 
                                                    exampleIndice]\
                                        ==\
                                       CLASS_LABELS[exampleIndice] \
                                    else 0 \
                                    for pastIterIndice in range(iterIndice)]))
    return fs

def updateCostmatrices(costMatrices, fs, iterIndice, NB_VIEW, DATASET_LENGTH, 
                        NB_CLASS, CLASS_LABELS):
    for viewIndice in range(NB_VIEW):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                if classe != CLASS_LABELS[exampleIndice]:
                    costMatrices[iterIndice+1, viewIndice, exampleIndice, classe] \
                    = math.exp(fs[iterIndice, viewIndice, exampleIndice, classe] - \
                      fs[iterIndice, viewIndice, 
                        exampleIndice, CLASS_LABELS[exampleIndice]])
                else:
                    costMatrices[iterIndice+1, viewIndice, exampleIndice, classe] \
                    = -1*np.sum(np.exp(fs[iterIndice, viewIndice, exampleIndice] - \
                        fs[iterIndice, viewIndice, exampleIndice, classe]))
    return costMatrices


def chooseView(predictions, generalCostMatrix, iterIndice, NB_VIEW, NB_CLASS, DATASET_LENGTH, CLASS_LABELS):
    edges = np.array([computeEdge(predictions[iterIndice, viewIndice], 
                                    generalCostMatrix[iterIndice], NB_CLASS, DATASET_LENGTH, CLASS_LABELS)
                      for viewIndice in range(NB_VIEW)])
    bestView = np.argmax(edges)
    return bestView, edges[bestView]


def updateGeneralFs(generalFs, iterIndice, predictions, alphas,
                    DATASET_LENGTH, NB_CLASS, bestView, generalAlphas, 
                    CLASS_LABELS):
    for exampleIndice in range(DATASET_LENGTH):
        for classe in range(NB_CLASS):
            generalFs[iterIndice, exampleIndice, classe]\
            = np.sum(np.array([generalAlphas[pastIterIndice]
                                if predictions[pastIterIndice, 
                                                bestView, 
                                                exampleIndice]
                                    ==
                                    CLASS_LABELS[exampleIndice]
                                 else 0
                                 for pastIterIndice in range(iterIndice)
                            ])
                    )
        return generalFs


def updateGeneralCostMatrix(generalCostMatrix, generalFs, iterIndice, 
                            DATASET_LENGTH, NB_CLASS, CLASS_LABELS):
    for exampleIndice in range(DATASET_LENGTH):
        for classe in range(NB_CLASS):
            if classe != CLASS_LABELS[exampleIndice]:
                generalCostMatrix[iterIndice, exampleIndice, classe]\
                = math.exp(generalFs[iterIndice, exampleIndice, classe] - \
                  generalFs[iterIndice, exampleIndice, CLASS_LABELS[exampleIndice]])
            else:
                generalCostMatrix[iterIndice, exampleIndice, classe]\
                = -1 * np.sum(np.exp(generalFs[iterIndice, exampleIndice] - \
                  generalFs[iterIndice, exampleIndice, classe]))
    return generalCostMatrix


def computeFinalFs(DATASET_LENGTH, NB_CLASS, generalAlphas, predictions, 
                    bestViews, CLASS_LABELS, NB_ITER):
    finalFs = np.zeros((DATASET_LENGTH, NB_CLASS))
    for exampleIndice in range(DATASET_LENGTH):
        for classe in range(NB_CLASS):
            finalFs[exampleIndice, classe] = np.sum(np.array([\
                                                        generalAlphas[iterIndice]\
                                                        if predictions[iterIndice, 
                                                                    bestViews[iterIndice],
                                                                    exampleIndice]\
                                                                ==\
                                                            CLASS_LABELS[exampleIndice]\
                                                        else 0 \
                                                        for iterIndice in range(NB_ITER)\
                                                    ])\
                                                )
    return finalFs


def trainMumbo(DATASET, CLASS_LABELS, NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH,
               classifierName, NB_CORES, classifierConfig):
    
    # Initialization
    costMatrices, \
    generalCostMatrix, fs, ds, edges, alphas, \
    predictions, generalAlphas, generalFs = initialize(NB_CLASS, NB_VIEW, 
                                                        NB_ITER, DATASET_LENGTH, 
                                                        CLASS_LABELS)
    bestViews = np.zeros(NB_ITER)
    bestClassifiers = []

    # Learning
    for iterIndice in range(NB_ITER):


        classifiers, predictedLabels = trainWeakClassifers(classifierName,
                                                           DATASET,
                                                           CLASS_LABELS,
                                                           costMatrices,
                                                           NB_CLASS,
                                                           DATASET_LENGTH,
                                                           iterIndice,
                                                           classifierConfig,
                                                           NB_CORES, NB_VIEW)

        predictions[iterIndice] = predictedLabels

        for viewIndice in range(NB_VIEW):

            edges[iterIndice, viewIndice] = computeEdge(predictions[iterIndice,
                                                                    viewIndice],
                                                        costMatrices[iterIndice+1,
                                                                     viewIndice], NB_CLASS, DATASET_LENGTH, CLASS_LABELS)

            alphas[iterIndice, viewIndice] = computeAlpha(edges[iterIndice,
                                                                viewIndice])

        ds = updateDs(ds, predictions, CLASS_LABELS, NB_VIEW, DATASET_LENGTH,
                      NB_CLASS, iterIndice)
        fs = updateFs(predictions, ds, alphas, fs, iterIndice, NB_VIEW,
                      DATASET_LENGTH, NB_CLASS, CLASS_LABELS)
        costMatrices = updateCostmatrices(costMatrices, fs, iterIndice,
                                          NB_VIEW, DATASET_LENGTH,
                                          NB_CLASS, CLASS_LABELS)

        bestView, edge = chooseView(predictions, generalCostMatrix,
                                    iterIndice, NB_VIEW, NB_CLASS, DATASET_LENGTH, CLASS_LABELS)

        bestViews[iterIndice] = bestView
        generalAlphas[iterIndice] = computeAlpha(edge)
        bestClassifiers.append(classifiers[bestView])
        generalFs = updateGeneralFs(generalFs, iterIndice, predictions, alphas,
                                    DATASET_LENGTH, NB_CLASS, bestView,
                                    generalAlphas, CLASS_LABELS)
        generalCostMatrix = updateGeneralCostMatrix(generalCostMatrix,
                                                    generalFs, iterIndice,
                                                    DATASET_LENGTH, NB_CLASS,
                                                    CLASS_LABELS)

    # finalFs = computeFinalFs(DATASET_LENGTH, NB_CLASS, generalAlphas, predictions, bestViews, CLASS_LABELS, NB_ITER)
    return bestClassifiers, generalAlphas, bestViews

def classifyMumbo(DATASET, classifiers, alphas, views, NB_CLASS):
    DATASET_LENGTH = len(DATASET[0])
    predictedLabels = np.zeros(DATASET_LENGTH)

    for exampleIndice in range(DATASET_LENGTH):
        votes = np.zeros(NB_CLASS)
        for classifier, alpha, view in zip(classifiers, alphas, views):
            data = np.array([np.array(DATASET[int(view)][exampleIndice])])
            votes[int(classifier.predict(data))]+=alpha
        predictedLabels[exampleIndice] = np.argmax(votes)
    return predictedLabels