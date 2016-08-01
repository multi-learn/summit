import numpy as np
import math
from joblib import Parallel, delayed
from Classifiers import *


# Data shape : ((Views, Examples, Corrdinates))


def initialize(NB_CLASS, NB_VIEW, NB_ITER, DATASET_LENGTH, CLASS_LABELS):
    costMatrices = np.array([
                                np.array([
                                             np.array([
                                                          np.array([1 if CLASS_LABELS[exampleIndice] != classe
                                                                    else -(NB_CLASS - 1)
                                                                    for classe in range(NB_CLASS)
                                                                    ]) for exampleIndice in range(DATASET_LENGTH)
                                                          ]) for viewIndice in range(NB_VIEW)])
                                    if iteration == 0
                                    else np.zeros((NB_VIEW, DATASET_LENGTH, NB_CLASS))
                                for iteration in range(NB_ITER + 1)
                                ])
    generalCostMatrix = np.array([
                                     np.array([
                                                  np.array([1 if CLASS_LABELS[exampleIndice] != classe
                                                            else -(NB_CLASS - 1)
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
    dist = dist - np.sum(np.array(
            [costMatrices[iterIndice, viewIndice, exampleIndice, CLASS_LABELS[exampleIndice]] for exampleIndice in
             range(DATASET_LENGTH)]))

    weights = np.array([-costMatrices[iterIndice, viewIndice,
                                      exampleIndice, CLASS_LABELS[exampleIndice]] / dist
                        for exampleIndice in range(DATASET_LENGTH)])
    return weights


def trainWeakClassifier(classifierName, monoviewDataset, CLASS_LABELS, costMatrices,
                        NB_CLASS, DATASET_LENGTH, iterIndice, viewIndice,
                        classifier_config):
    weights = computeWeights(costMatrices, NB_CLASS, DATASET_LENGTH,
                             iterIndice, viewIndice, CLASS_LABELS)
    classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
    classifierMethod = getattr(classifierModule, classifierName)
    classifier, classes, isBad, pTr = classifierMethod(monoviewDataset, CLASS_LABELS, classifier_config, weights)
    print "View : "+str(viewIndice)+" : "+str(np.mean(pTr))
    return classifier, classes, isBad


def trainWeakClassifers(classifierNames, DATASET, CLASS_LABELS, costMatrices,
                        NB_CLASS, DATASET_LENGTH, iterIndice, classifier_config,
                        NB_CORES, NB_VIEW):
    trainedClassifiers = []
    labelsMatrix = []
    areBad = []
    if NB_CORES > NB_VIEW:
        NB_JOBS = NB_VIEW
    else:
        NB_JOBS = NB_CORES
    trainedClassifiersAndLabels = Parallel(n_jobs=NB_JOBS)(
            delayed(trainWeakClassifier)(classifierNames[viewIndice], DATASET[viewIndice], CLASS_LABELS,
                                         costMatrices, NB_CLASS, DATASET_LENGTH,
                                         iterIndice, viewIndice, classifier_config[viewIndice])
            for viewIndice in range(NB_VIEW))

    for (classifier, labelsArray, isBad) in trainedClassifiersAndLabels:
        trainedClassifiers.append(classifier)
        labelsMatrix.append(labelsArray)
        areBad.append(isBad)
    return np.array(trainedClassifiers), np.array(labelsMatrix), np.array(areBad)


def computeEdge(predictionMatrix, costMatrix, NB_CLASS, DATASET_LENGTH, CLASS_LABELS):
    # return np.sum(np.array([np.sum(predictionMatrix*costMatrix[:,classIndice]) for classIndice in range(NB_CLASS)]))
    cCost = float(np.sum(np.array(
            [costMatrix[exampleIndice, int(predictionMatrix[exampleIndice])] for exampleIndice in range(DATASET_LENGTH)])))
    tCost = float(np.sum(
        np.array([-costMatrix[exampleIndice, CLASS_LABELS[exampleIndice]] for exampleIndice in range(DATASET_LENGTH)])))
    if tCost == 0.:
        edge = -cCost
    else:
        edge = -cCost / tCost
    return edge


def computeAlpha(edge):
    return 0.5 * math.log((1 + edge) / (1 - edge))


def allViewsClassifyWell(predictions, pastIterIndice, NB_VIEW, CLASS_LABEL,
                         exampleIndice):
    bool = True
    for viewIndice in range(NB_VIEW):
        if predictions[pastIterIndice, viewIndice, exampleIndice] != CLASS_LABEL:
            bool = False
    return bool


def updateDs(ds, predictions, CLASS_LABELS, NB_VIEW, DATASET_LENGTH, NB_CLASS,
             iterIndice):
    for viewIndice in range(NB_VIEW):
        for exampleIndice in range(DATASET_LENGTH):
            for pastIterIndice in range(iterIndice):

                if predictions[pastIterIndice, viewIndice, exampleIndice] \
                        == \
                        CLASS_LABELS[exampleIndice] \
                        or allViewsClassifyWell(predictions, pastIterIndice,
                                                NB_VIEW, CLASS_LABELS[exampleIndice],
                                                exampleIndice):

                    ds[pastIterIndice, viewIndice, exampleIndice] = 1
                else:
                    ds[pastIterIndice, viewIndice, exampleIndice] = 0
    return ds


def updateFs(predictions, ds, alphas, fs, iterIndice, NB_VIEW, DATASET_LENGTH,
             NB_CLASS, CLASS_LABELS):
    for viewIndice in range(NB_VIEW):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                fs[iterIndice, viewIndice, exampleIndice, classe] \
                    = np.sum(np.array([alphas[pastIterIndice, viewIndice]
                                       * ds[pastIterIndice, viewIndice, exampleIndice]
                                           if predictions[pastIterIndice, viewIndice,
                                                          exampleIndice]
                                              ==
                                              classe
                                           else 0
                                       for pastIterIndice in range(iterIndice)]))
    if np.amax(np.absolute(fs)) != 0:
        fs = fs/np.amax(np.absolute(fs))
    return fs


def updateCostmatrices(costMatrices, fs, iterIndice, NB_VIEW, DATASET_LENGTH,
                       NB_CLASS, CLASS_LABELS):
    for viewIndice in range(NB_VIEW):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                if classe != CLASS_LABELS[exampleIndice]:
                    costMatrices[iterIndice + 1, viewIndice, exampleIndice, classe] \
                        = 1.0*math.exp(fs[iterIndice, viewIndice, exampleIndice, classe] -
                                        fs[iterIndice, viewIndice, exampleIndice, CLASS_LABELS[exampleIndice]])
                else:
                    costMatrices[iterIndice + 1, viewIndice, exampleIndice, classe] \
                        = -1. * np.sum(np.exp(fs[iterIndice, viewIndice, exampleIndice] -
                                             fs[iterIndice, viewIndice, exampleIndice, classe]))
    costMatrices = costMatrices/np.amax(np.absolute(costMatrices))
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
            generalFs[iterIndice, exampleIndice, classe] \
                = np.sum(np.array([generalAlphas[pastIterIndice]
                                   if predictions[pastIterIndice,
                                                  bestView,
                                                  exampleIndice]
                                      ==
                                      classe
                                   else 0
                                   for pastIterIndice in range(iterIndice)
                                   ])
                         )
    if np.amax(np.absolute(generalFs)) != 0:
        generalFs = generalFs/np.amax(np.absolute(generalFs))
    return generalFs


def updateGeneralCostMatrix(generalCostMatrix, generalFs, iterIndice,
                            DATASET_LENGTH, NB_CLASS, CLASS_LABELS):
    for exampleIndice in range(DATASET_LENGTH):
        for classe in range(NB_CLASS):
            if classe != CLASS_LABELS[exampleIndice]:
                generalCostMatrix[iterIndice, exampleIndice, classe] \
                    = math.exp(generalFs[iterIndice, exampleIndice, classe] -
                               generalFs[iterIndice, exampleIndice, CLASS_LABELS[exampleIndice]])
            else:
                generalCostMatrix[iterIndice, exampleIndice, classe] \
                    = -1 * np.sum(np.exp(generalFs[iterIndice, exampleIndice] -
                                         generalFs[iterIndice, exampleIndice, classe]))
    # if np.amax(np.absolute(generalCostMatrix)) != 0:
    #     generalCostMatrix = generalCostMatrix/np.amax(np.absolute(generalCostMatrix))
    return generalCostMatrix


def train(DATASET, CLASS_LABELS, DATASET_LENGTH, NB_VIEW, NB_CLASS, NB_CORES,
          trainArguments):
    # Initialization
    classifierConfig, NB_ITER, classifierNames = trainArguments
    costMatrices, \
    generalCostMatrix, fs, ds, edges, alphas, \
    predictions, generalAlphas, generalFs = initialize(NB_CLASS, NB_VIEW,
                                                       NB_ITER, DATASET_LENGTH,
                                                       CLASS_LABELS)
    bestViews = np.zeros(NB_ITER)
    bestClassifiers = []

    # Learning
    for iterIndice in range(NB_ITER):
        print '_________________________________________________'
        classifiers, predictedLabels, areBad = trainWeakClassifers(classifierNames,
                                                           DATASET,
                                                           CLASS_LABELS,
                                                           costMatrices,
                                                           NB_CLASS,
                                                           DATASET_LENGTH,
                                                           iterIndice,
                                                           classifierConfig,
                                                           NB_CORES, NB_VIEW)
        if areBad.all():
            print "All bad for iteration " + str(iterIndice)

        predictions[iterIndice] = predictedLabels

        for viewIndice in range(NB_VIEW):
            edges[iterIndice, viewIndice] = computeEdge(predictions[iterIndice,
                                                                    viewIndice],
                                                        costMatrices[iterIndice,
                                                                     viewIndice], NB_CLASS, DATASET_LENGTH,
                                                        CLASS_LABELS)
            if areBad[viewIndice]:
                alphas[iterIndice, viewIndice] = 0.
            else:
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
        if areBad.all():
            generalAlphas[iterIndice] = 0.
        else:
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
    return (bestClassifiers, generalAlphas, bestViews)


def predict(DATASET, classifier, NB_CLASS):
    classifiers, alphas, views = classifier
    DATASET_LENGTH = len(DATASET[0])
    predictedLabels = np.zeros(DATASET_LENGTH)

    for exampleIndice in range(DATASET_LENGTH):
        votes = np.zeros(NB_CLASS)
        for classifier, alpha, view in zip(classifiers, alphas, views):
            data = DATASET[int(view)][exampleIndice]
            votes[int(classifier.predict(np.array([data])))] += alpha
        predictedLabels[exampleIndice] = np.argmax(votes)
    return predictedLabels


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




if __name__ == '__main__':
    from sklearn.metrics import classification_report
    from string import digits
    import os


    def extractRandomTrainingSet(DATA, CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_VIEW):
        nbTrainingExamples = int(DATASET_LENGTH * LEARNING_RATE)
        trainingExamplesIndices = np.random.random_integers(0, DATASET_LENGTH, nbTrainingExamples)
        trainData, trainLabels = [], []
        testData, testLabels = [], []
        for viewIndice in range(NB_VIEW):
            trainD, testD = [], []
            trainL, testL = [], []
            for i in np.arange(DATASET_LENGTH):
                if i in trainingExamplesIndices:
                    trainD.append(DATA[viewIndice][i])
                    trainL.append(CLASS_LABELS[i])
                else:
                    testD.append(DATA[viewIndice][i])
                    testL.append(CLASS_LABELS[i])
            trainData.append(np.array(trainD))
            testData.append(np.array(testD))
        trainLabels.append(np.array(trainL))
        testLabels.append(np.array(testL))
        return trainData, np.array(trainLabels[0]), testData, np.array(testLabels[0])

    def getAwaLabels(nbLabels, pathToAwa):
        file = open(pathToAwa + 'Animals_with_Attributes/classes.txt', 'U')
        linesFile = [''.join(line.strip().split()).translate(None, digits) for line in file.readlines()]
        awaLabels = [linesFile[label] for label in np.arange(nbLabels)]
        return awaLabels


    def getAwaData(pathToAwa, nbLabels, views):
        awaLabels = getAwaLabels(nbLabels, pathToAwa)
        nbView = len(views)
        labelDictionnary = {i: awaLabels[i] for i in np.arange(nbLabels)}
        viewDictionnary = {i: views[i] for i in np.arange(nbView)}
        rawData = []
        labels = []
        nbExample = 0
        # ij = []
        for view in np.arange(nbView):
            viewData = []
            for label in np.arange(nbLabels):
                pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionnary[view] + '/' + \
                                 labelDictionnary[label] + '/'
                examples = os.listdir(pathToExamples)
                if view == 0:
                    nbExample += len(examples)
                for example in examples:
                    exampleFile = open(pathToExamples + example)
                    viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in exampleFile][0])
                    if view == 0:
                        labels.append(label)
            rawData.append(np.array(viewData))
        data = rawData
        # data = np.empty((nbExample, nbView), dtype=list)
        # for viewIdice in np.arange(nbView):
        #     for exampleIndice in np.arange(nbExample):
        #         data[exampleIndice, viewIdice] = rawData[viewIdice][exampleIndice]
        #         # data[exampleIndice, viewIdice] = {i:rawData[viewIdice][exampleIndice][i] for i in np.arange(len(rawData[viewIdice][exampleIndice]))}

        return data, labels, viewDictionnary, labelDictionnary

    NB_CLASS = 5
    NB_ITER = 3
    classifierName="DecisionTree"
    NB_CORES = 3
    pathToAwa = "/home/doob/"
    views = ['phog-hist', 'decaf', 'cq-hist']
    NB_VIEW = len(views)
    LEARNING_RATE = 0.5
    classifierConfig = ['3']

    print "Getting db ..."
    DATASET, CLASS_LABELS, viewDictionnary, labelDictionnary = getAwaData(pathToAwa, NB_CLASS, views)
    target_names = labelDictionnary.values()
    # DATASET, CLASS_LABELS = DB.getDbfromCSV('/home/doob/OriginalData/')
    # NB_VIEW = 3
    CLASS_LABELS = np.array([int(label) for label in CLASS_LABELS])
    # print target_names
    # print labelDictionnary
    fullDatasetLength = len(CLASS_LABELS)

    trainData, trainLabels, testData, testLabels = extractRandomTrainingSet(DATASET, CLASS_LABELS, LEARNING_RATE,
                                                                               fullDatasetLength, NB_VIEW)
    DATASET_LENGTH = len(trainLabels)
    # print len(trainData), trainData[0].shape, len(trainLabels)
    print "Done."

    print 'Training Mumbo ...'
    trainArguments = classifierConfig, NB_ITER, classifierName

    bestClassifiers, generalAlphas, bestViews = train(trainData, trainLabels, DATASET_LENGTH, NB_VIEW, NB_CLASS, NB_CORES,
                                                      trainArguments)
    # DATASET, VIEW_DIMENSIONS, CLASS_LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
    print "Trained."

    print "Predicting ..."
    predictedTrainLabels = predict(trainData, (bestClassifiers, generalAlphas, bestViews), NB_CLASS)
    predictedTestLabels = predict(testData, (bestClassifiers, generalAlphas, bestViews), NB_CLASS)
    print 'Done.'
    print 'Reporting ...'
    predictedTrainLabelsByIter = classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    predictedTestLabelsByIter = classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
    print str(NB_VIEW)+" views, "+str(NB_CLASS)+" classes, "+str(classifierConfig)+" depth trees"
    print "Best views = "+str(bestViews)
    print "Is equal : "+str((predictedTrainLabels==predictedTrainLabelsByIter[NB_ITER-1]).all())

    print "On train : "
    print classification_report(trainLabels, predictedTrainLabels, target_names=target_names)
    print "On test : "
    print classification_report(testLabels, predictedTestLabels, target_names=target_names)