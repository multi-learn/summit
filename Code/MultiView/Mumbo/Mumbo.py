import numpy as np
import math
from joblib import Parallel, delayed
from Classifiers import *
import logging


# Data shape : ((Views, Examples, Corrdinates))

def computeWeights(DATASET_LENGTH, iterIndex, viewIndice, CLASS_LABELS, costMatrices):
    dist = np.sum(costMatrices[iterIndex, viewIndice])
    dist = dist - np.sum(np.array(
        [costMatrices[iterIndex, viewIndice, exampleIndice, CLASS_LABELS[exampleIndice]] for exampleIndice in
         range(DATASET_LENGTH)]))

    weights = np.array([-costMatrices[iterIndex, viewIndice,
                                      exampleIndice, CLASS_LABELS[exampleIndice]] / dist
                        for exampleIndice in range(DATASET_LENGTH)])
    return weights

def trainWeakClassifier(classifierName, monoviewDataset, CLASS_LABELS,
                        DATASET_LENGTH, viewIndice, classifier_config, iterIndex, costMatrices):
    weights = computeWeights(DATASET_LENGTH, iterIndex, viewIndice, CLASS_LABELS, costMatrices)
    classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
    classifierMethod = getattr(classifierModule, classifierName)
    classifier, classes, isBad, pTr = classifierMethod(monoviewDataset, CLASS_LABELS, classifier_config, weights)
    logging.debug("\t\t\tView " + str(viewIndice) + " : " + str(np.mean(pTr)))
    return classifier, classes, isBad

def trainWeakClassifier_hdf5(classifierName, monoviewDataset, CLASS_LABELS, DATASET_LENGTH,
                             viewIndice, classifier_config, viewName, iterIndex, costMatrices):
    weights = computeWeights(DATASET_LENGTH, iterIndex, viewIndice, CLASS_LABELS, costMatrices)
    classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
    classifierMethod = getattr(classifierModule, classifierName)
    classifier, classes, isBad, pTr = classifierMethod(monoviewDataset, CLASS_LABELS, classifier_config, weights)
    logging.debug("\t\t\tFor " + viewName + " : " + str(np.mean(pTr)) +" : "+ str(not isBad))
    return classifier, classes, isBad



class Mumbo:

    def __init__(self, NB_VIEW, DATASET_LENGTH, CLASS_LABELS, NB_CORES=1,**kwargs):
        self.nbIter = kwargs["NB_ITER"]
        self.classifiersNames = kwargs["classifiersNames"]
        self.classifiersConfigs = kwargs["classifiersConfigs"]
        nbClass = len(set(CLASS_LABELS))
        self.nbIter = kwargs["NB_ITER"]
        self.costMatrices = np.array([
                                        np.array([
                                                     np.array([
                                                                  np.array([1 if CLASS_LABELS[exampleIndice] != classe
                                                                            else -(nbClass - 1)
                                                                            for classe in range(nbClass)
                                                                            ]) for exampleIndice in range(DATASET_LENGTH)
                                                                  ]) for viewIndice in range(NB_VIEW)])
                                        if iteration == 0
                                        else np.zeros((NB_VIEW, DATASET_LENGTH, nbClass))
                                        for iteration in range(self.nbIter + 1)
                                        ])
        self.generalCostMatrix = np.array([
                                              np.array([
                                                           np.array([1 if CLASS_LABELS[exampleIndice] != classe
                                                                     else -(nbClass - 1)
                                                                     for classe in range(nbClass)
                                                                     ]) for exampleIndice in range(DATASET_LENGTH)
                                                           ]) for iteration in range(self.nbIter)
                                              ])
        self.fs = np.zeros((self.nbIter, NB_VIEW, DATASET_LENGTH, nbClass))
        self.ds = np.zeros((self.nbIter, NB_VIEW, DATASET_LENGTH))
        self.edges = np.zeros((self.nbIter, NB_VIEW))
        self.alphas = np.zeros((self.nbIter, NB_VIEW))
        self.predictions = np.zeros((self.nbIter, NB_VIEW, DATASET_LENGTH))
        self.generalAlphas = np.zeros(self.nbIter)
        self.generalFs = np.zeros((self.nbIter, DATASET_LENGTH, nbClass))
        self.nbCores = NB_CORES
        self.iterIndex = 0
        self.bestClassifiers = []
        self.bestViews = np.zeros(self.nbIter, dtype=int)
        # costMatrices = np.array([
        #                             np.array([
        #                                          np.array([
        #                                                       np.array([1 if CLASS_LABELS[exampleIndice] != classe
        #                                                                 else -(NB_CLASS - 1)
        #                                                                 for classe in range(NB_CLASS)
        #                                                                 ]) for exampleIndice in range(DATASET_LENGTH)
        #                                                       ]) for viewIndice in range(NB_VIEW)])
        #                             if iteration == 0
        #                             else np.zeros((NB_VIEW, DATASET_LENGTH, NB_CLASS))
        #                             for iteration in range(NB_ITER + 1)
        #                             ])
        # generalCostMatrix = np.array([
        #                                  np.array([
        #                                               np.array([1 if CLASS_LABELS[exampleIndice] != classe
        #                                                         else -(NB_CLASS - 1)
        #                                                         for classe in range(NB_CLASS)
        #                                                         ]) for exampleIndice in range(DATASET_LENGTH)
        #                                               ]) for iteration in range(NB_ITER)
        #                                  ])
        # fs = np.zeros((NB_ITER, NB_VIEW, DATASET_LENGTH, NB_CLASS))
        # ds = np.zeros((NB_ITER, NB_VIEW, DATASET_LENGTH))
        # edges = np.zeros((NB_ITER, NB_VIEW))
        # alphas = np.zeros((NB_ITER, NB_VIEW))
        # predictions = np.zeros((NB_ITER, NB_VIEW, DATASET_LENGTH))
        # generalAlphas = np.zeros(NB_ITER)
        # generalFs = np.zeros((NB_ITER, DATASET_LENGTH, NB_CLASS))
        # return costMatrices, generalCostMatrix, fs, ds, edges, alphas, \
        #        predictions, generalAlphas, generalFs



    def trainWeakClassifiers(self, DATASET, CLASS_LABELS, NB_CLASS, DATASET_LENGTH, NB_VIEW):
        trainedClassifiers = []
        labelsMatrix = []
        areBad = []
        if self.nbCores > NB_VIEW:
            NB_JOBS = NB_VIEW
        else:
            NB_JOBS = self.nbCores
        classifiersConfigs = self.classifiersConfigs
        costMatrices = self.costMatrices
        classifiersNames = self.classifiersNames
        iterIndex = self.iterIndex
        trainedClassifiersAndLabels = Parallel(n_jobs=NB_JOBS)(
                delayed(trainWeakClassifier)(classifiersNames[viewIndice], DATASET[viewIndice], CLASS_LABELS,
                                             DATASET_LENGTH, viewIndice, classifiersConfigs[viewIndice], iterIndex,
                                             costMatrices)
                for viewIndice in range(NB_VIEW))

        for (classifier, labelsArray, isBad) in trainedClassifiersAndLabels:
            trainedClassifiers.append(classifier)
            labelsMatrix.append(labelsArray)
            areBad.append(isBad)
        return np.array(trainedClassifiers), np.array(labelsMatrix), np.array(areBad)

    def trainWeakClassifiers_hdf5(self, DATASET, trainIndices, NB_CLASS,
                                 DATASET_LENGTH, NB_VIEW):
        trainedClassifiers = []
        labelsMatrix = []
        areBad = []
        if self.nbCores > NB_VIEW:
            NB_JOBS = NB_VIEW
        else:
            NB_JOBS = self.nbCores
        classifiersConfigs = self.classifiersConfigs
        costMatrices = self.costMatrices
        classifiersNames = self.classifiersNames
        iterIndex = self.iterIndex
        trainedClassifiersAndLabels = Parallel(n_jobs=NB_JOBS)(
                delayed(trainWeakClassifier_hdf5)(classifiersNames[viewIndex],
                                             DATASET["/View"+str(viewIndex)+"/matrix"][trainIndices, :],
                                             DATASET["/Labels/labelsArray"][trainIndices],
                                             DATASET_LENGTH,
                                             viewIndex, classifiersConfigs[viewIndex],
                                             str(DATASET["/View"+str(viewIndex)+"/name"][...]), iterIndex, costMatrices)
                for viewIndex in range(NB_VIEW))

        for (classifier, labelsArray, isBad) in trainedClassifiersAndLabels:
            trainedClassifiers.append(classifier)
            labelsMatrix.append(labelsArray)
            areBad.append(isBad)
        return np.array(trainedClassifiers), np.array(labelsMatrix), np.array(areBad)

    def computeEdge(self, viewIndex, DATASET_LENGTH, CLASS_LABELS):
        predictionMatrix = self.predictions[self.iterIndex, viewIndex]
        costMatrix = self.costMatrices[self.iterIndex, viewIndex]
        # return np.sum(np.array([np.sum(predictionMatrix*costMatrix[:,classIndice]) for classIndice in range(NB_CLASS)]))
        cCost = float(np.sum(np.array(
                [costMatrix[exampleIndice, int(predictionMatrix[exampleIndice])] for exampleIndice in
                 range(DATASET_LENGTH)])))
        tCost = float(np.sum(
                np.array([-costMatrix[exampleIndice, CLASS_LABELS[exampleIndice]] for exampleIndice in
                          range(DATASET_LENGTH)])))
        if tCost == 0.:
            self.edges[self.iterIndex, viewIndex] = -cCost
        else:
            self.edges[self.iterIndex, viewIndex] = -cCost / tCost


    def computeAlpha(self, edge):
        if edge < 1 :
            return 0.5 * math.log((1 + edge) / (1 - edge))
        else:
            return 0

    def allViewsClassifyWell(self, predictions, pastIterIndice, NB_VIEW, CLASS_LABEL, exampleIndice):
        boolean = True
        for viewIndice in range(NB_VIEW):
            if predictions[pastIterIndice, viewIndice, exampleIndice] != CLASS_LABEL:
                boolean = False
        return boolean

    def updateDs(self, CLASS_LABELS, NB_VIEW, DATASET_LENGTH):
        for viewIndice in range(NB_VIEW):
            for exampleIndice in range(DATASET_LENGTH):
                for pastIterIndice in range(self.iterIndex):

                    if self.predictions[pastIterIndice, viewIndice, exampleIndice] \
                            == \
                            CLASS_LABELS[exampleIndice] \
                            or self.allViewsClassifyWell(self.predictions, pastIterIndice,
                                                    NB_VIEW, CLASS_LABELS[exampleIndice],
                                                    exampleIndice):

                        self.ds[pastIterIndice, viewIndice, exampleIndice] = 1
                    else:
                        self.ds[pastIterIndice, viewIndice, exampleIndice] = 0
        #return ds

    def updateFs(self, NB_VIEW, DATASET_LENGTH, NB_CLASS):
        for viewIndice in range(NB_VIEW):
            for exampleIndice in range(DATASET_LENGTH):
                for classe in range(NB_CLASS):
                    self.fs[self.iterIndex, viewIndice, exampleIndice, classe] \
                        = np.sum(np.array([self.alphas[pastIterIndice, viewIndice]
                                           * self.ds[pastIterIndice, viewIndice, exampleIndice]
                                           if self.predictions[pastIterIndice, viewIndice,
                                                          exampleIndice]
                                              ==
                                              classe
                                           else 0
                                           for pastIterIndice in range(self.iterIndex)]))
        if np.amax(np.absolute(self.fs)) != 0:
            self.fs /= np.amax(np.absolute(self.fs))
        #return fs

    def updateCostmatrices(self, NB_VIEW, DATASET_LENGTH, NB_CLASS, CLASS_LABELS):
        for viewIndice in range(NB_VIEW):
            for exampleIndice in range(DATASET_LENGTH):
                for classe in range(NB_CLASS):
                    if classe != CLASS_LABELS[exampleIndice]:
                        self.costMatrices[self.iterIndex + 1, viewIndice, exampleIndice, classe] \
                            = 1.0 * math.exp(self.fs[self.iterIndex, viewIndice, exampleIndice, classe] -
                                             self.fs[self.iterIndex, viewIndice, exampleIndice, CLASS_LABELS[exampleIndice]])
                    else:
                        self.costMatrices[self.iterIndex + 1, viewIndice, exampleIndice, classe] \
                            = -1. * np.sum(np.exp(self.fs[self.iterIndex, viewIndice, exampleIndice] -
                                                  self.fs[self.iterIndex, viewIndice, exampleIndice, classe]))
        self.costMatrices /= np.amax(np.absolute(self.costMatrices))
        #return costMatrices

    def chooseView(self, NB_VIEW, CLASS_LABELS, DATASET_LENGTH):
        for viewIndex in range(NB_VIEW):
            self.computeEdge(viewIndex, DATASET_LENGTH, CLASS_LABELS)

        bestView = np.argmax(self.edges[self.iterIndex, :])
        return bestView, self.edges[self.iterIndex, bestView]

    def updateGeneralFs(self, DATASET_LENGTH, NB_CLASS, bestView):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                self.generalFs[self.iterIndex, exampleIndice, classe] \
                    = np.sum(np.array([self.generalAlphas[pastIterIndice]
                                       if self.predictions[pastIterIndice,
                                                      bestView,
                                                      exampleIndice]
                                          ==
                                          classe
                                       else 0
                                       for pastIterIndice in range(self.iterIndex)
                                       ])
                             )
        if np.amax(np.absolute(self.generalFs)) != 0:
            self.generalFs /= np.amax(np.absolute(self.generalFs))
        #return generalFs

    def updateGeneralCostMatrix(self, DATASET_LENGTH, NB_CLASS, CLASS_LABELS):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                if classe != CLASS_LABELS[exampleIndice]:
                    self.generalCostMatrix[self.iterIndex, exampleIndice, classe] \
                        = math.exp(self.generalFs[self.iterIndex, exampleIndice, classe] -
                                   self.generalFs[self.iterIndex, exampleIndice, CLASS_LABELS[exampleIndice]])
                else:
                    self.generalCostMatrix[self.iterIndex, exampleIndice, classe] \
                        = -1 * np.sum(np.exp(self.generalFs[self.iterIndex, exampleIndice] -
                                             self.generalFs[self.iterIndex, exampleIndice, classe]))
        # if np.amax(np.absolute(generalCostMatrix)) != 0:
        #     generalCostMatrix = generalCostMatrix/np.amax(np.absolute(generalCostMatrix))

    def fit(self, DATASET, CLASS_LABELS, **kwargs):
        # Initialization
        DATASET_LENGTH = len(CLASS_LABELS)
        NB_VIEW = len(DATASET)
        NB_CLASS = len(set(CLASS_LABELS))
        # costMatrices, \
        # generalCostMatrix, fs, ds, edges, alphas, \
        # predictions, generalAlphas, generalFs = initialize(NB_CLASS, NB_VIEW,
        #                                                    NB_ITER, DATASET_LENGTH,
        #                                                    CLASS_LABELS)
        bestViews = np.zeros(self.nbIter)
        bestClassifiers = []

        # Learning
        for i in range(self.nbIter):
            logging.debug('\t\tStart:\t Iteration ' + str(self.iterIndex + 1))
            classifiers, predictedLabels, areBad = self.trainWeakClassifiers(DATASET, CLASS_LABELS, NB_CLASS,
                                                                             DATASET_LENGTH, NB_VIEW)
            if areBad.all():
                logging.warning("All bad for iteration " + str(self.iterIndex))

            self.predictions[self.iterIndex] = predictedLabels

            for viewIndice in range(NB_VIEW):
                self.computeEdge(viewIndice, DATASET_LENGTH, CLASS_LABELS)
                if areBad[viewIndice]:
                    self.alphas[self.iterIndex, viewIndice] = 0.
                else:
                    self.alphas[self.iterIndex, viewIndice] = self.computeAlpha(self.edges[self.iterIndex,
                                                                        viewIndice])
            self.updateDs(CLASS_LABELS, NB_VIEW, DATASET_LENGTH)
            self.updateFs(NB_VIEW, DATASET_LENGTH, NB_CLASS)

            self.updateCostmatrices(NB_VIEW, DATASET_LENGTH, NB_CLASS, CLASS_LABELS)
            bestView, edge = self.chooseView(NB_VIEW, CLASS_LABELS, DATASET_LENGTH)
            self.bestViews[self.iterIndex] = bestView
            if areBad.all():
                self.generalAlphas[self.iterIndex] = 0.
            else:
                self.generalAlphas[self.iterIndex] = self.computeAlpha(edge)
            self.bestClassifiers.append(classifiers[bestView])
            self.updateGeneralFs(DATASET_LENGTH, NB_CLASS, bestView)
            self.updateGeneralCostMatrix(DATASET_LENGTH, NB_CLASS, CLASS_LABELS)

        # finalFs = computeFinalFs(DATASET_LENGTH, NB_CLASS, generalAlphas, predictions, bestViews, LABELS, NB_ITER)

    def fit_hdf5(self, DATASET, trainIndices=None):
        # Initialization
        if not trainIndices:
            trainIndices = range(DATASET.get("datasetLength").value)
        NB_CLASS = DATASET["/nbClass"][...]
        NB_VIEW = DATASET["/nbView"][...]
        DATASET_LENGTH = len(trainIndices)
        LABELS = DATASET["/Labels/labelsArray"][trainIndices]
        # costMatrices, \
        # generalCostMatrix, fs, ds, edges, alphas, \
        # predictions, generalAlphas, generalFs = initialize(NB_CLASS, NB_VIEW,
        #                                                    NB_ITER, DATASET_LENGTH,
        #                                                    LABELS[trainIndices])
        bestViews = np.zeros(self.nbIter)
        bestClassifiers = []

        # Learning
        self.iterIndex = 0
        for i in range(self.nbIter):
            logging.debug('\t\tStart:\t Iteration ' + str(self.iterIndex + 1))
            classifiers, predictedLabels, areBad = self.trainWeakClassifiers_hdf5(DATASET, trainIndices, NB_CLASS,
                                                                                  DATASET_LENGTH, NB_VIEW)
            if areBad.all():
                logging.warning("All bad for iteration " + str(self.iterIndex))

            self.predictions[self.iterIndex] = predictedLabels

            for viewIndice in range(NB_VIEW):
                self.computeEdge(viewIndice, DATASET_LENGTH, LABELS)
                if areBad[viewIndice]:
                    self.alphas[self.iterIndex, viewIndice] = 0.
                else:
                    self.alphas[self.iterIndex, viewIndice] = self.computeAlpha(self.edges[self.iterIndex, viewIndice])

            self.updateDs(LABELS, NB_VIEW, DATASET_LENGTH)
            self.updateFs(NB_VIEW, DATASET_LENGTH, NB_CLASS)

            self.updateCostmatrices(NB_VIEW, DATASET_LENGTH, NB_CLASS, LABELS)
            bestView, edge = self.chooseView(NB_VIEW, LABELS, DATASET_LENGTH)
            self.bestViews[self.iterIndex] = bestView
            if areBad.all():
                self.generalAlphas[self.iterIndex] = 0.
            else:
                self.generalAlphas[self.iterIndex] = self.computeAlpha(edge)
            self.bestClassifiers.append(classifiers[bestView])
            self.updateGeneralFs(DATASET_LENGTH, NB_CLASS, bestView)
            self.updateGeneralCostMatrix(DATASET_LENGTH, NB_CLASS,LABELS)
            self.iterIndex += 1

        # finalFs = computeFinalFs(DATASET_LENGTH, NB_CLASS, generalAlphas, predictions, bestViews, LABELS, NB_ITER)

    def predict(self, DATASET, NB_CLASS=2):
        DATASET_LENGTH = len(DATASET[0])
        predictedLabels = np.zeros(DATASET_LENGTH)

        for exampleIndice in range(DATASET_LENGTH):
            votes = np.zeros(NB_CLASS)
            for classifier, alpha, view in zip(self.bestClassifiers, self.alphas, self.bestViews):
                data = DATASET[int(view)][exampleIndice]
                votes[int(classifier.predict(np.array([data])))] += alpha
            predictedLabels[exampleIndice] = np.argmax(votes)
        return predictedLabels

    def predict_hdf5(self, DATASET, usedIndices=None):
        NB_CLASS = DATASET.get("nbClass").value
        if usedIndices == None:
            usedIndices = range(DATASET.get("datasetLength").value)
        if usedIndices:
            DATASET_LENGTH = len(usedIndices)
            predictedLabels = np.zeros(DATASET_LENGTH)

            for labelIndex, exampleIndex in enumerate(usedIndices):
                votes = np.zeros(NB_CLASS)
                for classifier, alpha, view in zip(self.bestClassifiers, self.alphas, self.bestViews):
                    data = DATASET["/View"+str(int(view))+"/matrix"][exampleIndex, :]
                    votes[int(classifier.predict(np.array([data])))] += alpha[view]
                predictedLabels[labelIndex] = np.argmax(votes)
        else:
            predictedLabels = []
        return predictedLabels

    def classifyMumbobyIter(self, DATASET, NB_CLASS=2):
        DATASET_LENGTH = len(DATASET[0])
        NB_ITER = len(self.bestClassifiers)
        predictedLabels = np.zeros((DATASET_LENGTH, NB_ITER))
        votes = np.zeros((DATASET_LENGTH, NB_CLASS))

        for classifier, alpha, view, iterIndice in zip(self.bestClassifiers, self.alphas, self.bestViews, range(NB_ITER)):
            votesByIter = np.zeros((DATASET_LENGTH, NB_CLASS))

            for exampleIndice in range(DATASET_LENGTH):
                data = np.array([np.array(DATASET[int(view)][exampleIndice])])
                votesByIter[exampleIndice, int(self.predict(data, NB_CLASS))] += alpha
                votes[exampleIndice] = votes[exampleIndice] + np.array(votesByIter[exampleIndice])
                predictedLabels[exampleIndice, iterIndice] = np.argmax(votes[exampleIndice])

        return np.transpose(predictedLabels)

    def classifyMumbobyIter_hdf5(self, DATASET, usedIndices=None, NB_CLASS=2):
        if usedIndices == None:
            usedIndices = range(DATASET.get("datasetLength").value)
        if usedIndices:
            DATASET_LENGTH = len(usedIndices)
            predictedLabels = np.zeros((DATASET_LENGTH, self.nbIter))
            votes = np.zeros((DATASET_LENGTH, NB_CLASS))

            for iterIndex, (classifier, alpha, view) in enumerate(zip(self.bestClassifiers, self.alphas, self.bestViews)):
                votesByIter = np.zeros((DATASET_LENGTH, NB_CLASS))

                for usedExampleIndex, exampleIndex in enumerate(usedIndices):
                    data = np.array([np.array(DATASET["/View" + str(int(view)) + "/matrix"][exampleIndex, :])])
                    votesByIter[usedExampleIndex, int(classifier.predict(data))] += alpha[view]
                    votes[usedExampleIndex] = votes[usedExampleIndex] + np.array(votesByIter[usedExampleIndex])
                    predictedLabels[usedExampleIndex, iterIndex] = np.argmax(votes[usedExampleIndex])
        else:
            predictedLabels = []
            for i in range(self.nbIter):
                predictedLabels.append([])

        return np.transpose(predictedLabels)
#
# if __name__ == '__main__':
#     from sklearn.metrics import classification_report
#     from string import digits
#     import os
#
#
#     def extractRandomTrainingSet(DATA, CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_VIEW):
#         nbTrainingExamples = int(DATASET_LENGTH * LEARNING_RATE)
#         trainingExamplesIndices = np.random.random_integers(0, DATASET_LENGTH, nbTrainingExamples)
#         trainData, trainLabels = [], []
#         testData, testLabels = [], []
#         for viewIndice in range(NB_VIEW):
#             trainD, testD = [], []
#             trainL, testL = [], []
#             for i in np.arange(DATASET_LENGTH):
#                 if i in trainingExamplesIndices:
#                     trainD.append(DATA[viewIndice][i])
#                     trainL.append(CLASS_LABELS[i])
#                 else:
#                     testD.append(DATA[viewIndice][i])
#                     testL.append(CLASS_LABELS[i])
#             trainData.append(np.array(trainD))
#             testData.append(np.array(testD))
#         trainLabels.append(np.array(trainL))
#         testLabels.append(np.array(testL))
#         return trainData, np.array(trainLabels[0]), testData, np.array(testLabels[0])
#
#
#     def getAwaLabels(nbLabels, pathToAwa):
#         file = open(pathToAwa + 'Animals_with_Attributes/classes.txt', 'U')
#         linesFile = [''.join(line.strip().split()).translate(None, digits) for line in file.readlines()]
#         awaLabels = [linesFile[label] for label in np.arange(nbLabels)]
#         return awaLabels
#
#
#     def getAwaData(pathToAwa, nbLabels, views):
#         awaLabels = getAwaLabels(nbLabels, pathToAwa)
#         nbView = len(views)
#         labelDictionnary = {i: awaLabels[i] for i in np.arange(nbLabels)}
#         viewDictionnary = {i: views[i] for i in np.arange(nbView)}
#         rawData = []
#         labels = []
#         nbExample = 0
#         # ij = []
#         for view in np.arange(nbView):
#             viewData = []
#             for label in np.arange(nbLabels):
#                 pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionnary[view] + '/' + \
#                                  labelDictionnary[label] + '/'
#                 examples = os.listdir(pathToExamples)
#                 if view == 0:
#                     nbExample += len(examples)
#                 for example in examples:
#                     exampleFile = open(pathToExamples + example)
#                     viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in exampleFile][0])
#                     if view == 0:
#                         labels.append(label)
#             rawData.append(np.array(viewData))
#         data = rawData
#         # data = np.empty((nbExample, nbView), dtype=list)
#         # for viewIdice in np.arange(nbView):
#         #     for exampleIndice in np.arange(nbExample):
#         #         data[exampleIndice, viewIdice] = rawData[viewIdice][exampleIndice]
#         #         # data[exampleIndice, viewIdice] = {i:rawData[viewIdice][exampleIndice][i] for i in np.arange(len(rawData[viewIdice][exampleIndice]))}
#
#         return data, labels, viewDictionnary, labelDictionnary
#
#
#     NB_CLASS = 5
#     NB_ITER = 3
#     classifierName = "DecisionTree"
#     NB_CORES = 3
#     pathToAwa = "/home/doob/"
#     views = ['phog-hist', 'decaf', 'cq-hist']
#     NB_VIEW = len(views)
#     LEARNING_RATE = 0.5
#     classifierConfig = ['3']
#
#     print "Getting db ..."
#     DATASET, CLASS_LABELS, viewDictionnary, labelDictionnary = getAwaData(pathToAwa, NB_CLASS, views)
#     target_names = labelDictionnary.values()
#     # DATASET, LABELS = DB.getDbfromCSV('/home/doob/OriginalData/')
#     # NB_VIEW = 3
#     CLASS_LABELS = np.array([int(label) for label in CLASS_LABELS])
#     # print target_names
#     # print labelDictionnary
#     fullDatasetLength = len(CLASS_LABELS)
#
#     trainData, trainLabels, testData, testLabels = extractRandomTrainingSet(DATASET, CLASS_LABELS, LEARNING_RATE,
#                                                                             fullDatasetLength, NB_VIEW)
#     DATASET_LENGTH = len(trainLabels)
#     # print len(trainData), trainData[0].shape, len(trainLabels)
#     print "Done."
#
#     print 'Training Mumbo ...'
#     trainArguments = classifierConfig, NB_ITER, classifierName
#
#     bestClassifiers, generalAlphas, bestViews = train(trainData, trainLabels, DATASET_LENGTH, NB_VIEW, NB_CLASS,
#                                                       NB_CORES,
#                                                       trainArguments)
#     # DATASET, VIEW_DIMENSIONS, LABELS = DB.createFakeData(NB_VIEW, DATASET_LENGTH, NB_CLASS)
#     print "Trained."
#
#     print "Predicting ..."
#     predictedTrainLabels = predict(trainData, (bestClassifiers, generalAlphas, bestViews), NB_CLASS)
#     predictedTestLabels = predict(testData, (bestClassifiers, generalAlphas, bestViews), NB_CLASS)
#     print 'Done.'
#     print 'Reporting ...'
#     predictedTrainLabelsByIter = classifyMumbobyIter(trainData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
#     predictedTestLabelsByIter = classifyMumbobyIter(testData, bestClassifiers, generalAlphas, bestViews, NB_CLASS)
#     print str(NB_VIEW) + " views, " + str(NB_CLASS) + " classes, " + str(classifierConfig) + " depth trees"
#     print "Best views = " + str(bestViews)
#     print "Is equal : " + str((predictedTrainLabels == predictedTrainLabelsByIter[NB_ITER - 1]).all())
#
#     print "On train : "
#     print classification_report(trainLabels, predictedTrainLabels, target_names=target_names)
#     print "On test : "
#     print classification_report(testLabels, predictedTestLabels, target_names=target_names)
