import numpy as np
import math
from joblib import Parallel, delayed
import itertools
from Classifiers import *
import time
import pkgutil
import logging
from sklearn.metrics import accuracy_score
from utils.Dataset import getV


# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


# Data shape : ((Views, Examples, Corrdinates))

def getBenchmark(benchmark, args=None):
    allAlgos = [name for _, name, isPackage in
                pkgutil.iter_modules(['Multiview/Mumbo/Classifiers'])
                if not isPackage and not name in ["SubSampling", "ModifiedMulticlass", "Kover"]]
    if args is None or args.MU_types != ['']:
        benchmark["Multiview"]["Mumbo"] = allAlgos
    else:
        benchmark["Multiview"]["Mumbo"] = args.MU_types
    return benchmark


def getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices):
    argumentsList = []

    arguments = {"CL_type": "Mumbo",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(args.CL_classes),
                 "LABELS_NAMES": args.CL_classes,
                 "MumboKWARGS": {"classifiersNames": args.MU_types,
                                 "maxIter": int(args.MU_iter[0]), "minIter": int(args.MU_iter[1]),
                                 "threshold": args.MU_iter[2],
                                 "classifiersConfigs": [map(float, argument.split(":")) for argument in
                                                        args.MU_config], "nbView": (len(viewsIndices))}}
    argumentsList.append(arguments)
    return argumentsList


def computeWeights(DATASET_LENGTH, iterIndex, viewIndice, CLASS_LABELS, costMatrices):
    dist = np.sum(costMatrices[iterIndex, viewIndice])
    dist = dist - np.sum(np.array(
        [costMatrices[iterIndex, viewIndice, exampleIndice, int(CLASS_LABELS[exampleIndice])] for exampleIndice in
         range(DATASET_LENGTH)]))

    weights = np.array([-costMatrices[iterIndex, viewIndice,
                                      exampleIndice, int(CLASS_LABELS[exampleIndice])] / dist
                        for exampleIndice in range(DATASET_LENGTH)])
    return weights


def trainWeakClassifier(classifierName, monoviewDataset, CLASS_LABELS,
                        DATASET_LENGTH, viewIndice, classifier_config, iterIndex, costMatrices):
    weights = computeWeights(DATASET_LENGTH, iterIndex, viewIndice, CLASS_LABELS, costMatrices)
    classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
    classifierMethod = getattr(classifierModule, classifierName)
    classifier, classes, isBad, averageAccuracy = classifierMethod(monoviewDataset, CLASS_LABELS, classifier_config, weights)
    logging.debug("\t\t\tView " + str(viewIndice) + " : " + str(averageAccuracy))
    return classifier, classes, isBad, averageAccuracy


def trainWeakClassifier_hdf5(classifierName, monoviewDataset, CLASS_LABELS, DATASET_LENGTH,
                             viewIndice, classifier_config, viewName, iterIndex, costMatrices, classifierIndex, randomState):
    weights = computeWeights(DATASET_LENGTH, iterIndex, classifierIndex, CLASS_LABELS, costMatrices)
    classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
    classifierMethod = getattr(classifierModule, classifierName)
    classifier, classes, isBad, averageAccuracy = classifierMethod(monoviewDataset, CLASS_LABELS, classifier_config, weights, randomState)
    logging.debug("\t\t\tView " + str(viewIndice) + " : " + str(averageAccuracy))
    return classifier, classes, isBad, averageAccuracy


def gridSearch_hdf5(DATASET, viewIndices, classificationKWARGS, learningIndices, randomState, metric=None, nIter=None):
    classifiersNames = classificationKWARGS["classifiersNames"]
    bestSettings = []
    for classifierIndex, classifierName in enumerate(classifiersNames):
        logging.debug("\tStart:\t Random search for "+classifierName+" on "+DATASET.get("View"+str(viewIndices[classifierIndex])).attrs["name"])
        classifierModule = globals()[classifierName]  # Permet d'appeler une fonction avec une string
        classifierGridSearch = getattr(classifierModule, "hyperParamSearch")
        bestSettings.append(classifierGridSearch(getV(DATASET, viewIndices[classifierIndex], learningIndices),
                                             DATASET.get("Labels").value[learningIndices], randomState, metric=metric))
        logging.debug("\tDone:\t Gridsearch for "+classifierName)
    return bestSettings, None


def getCLString(classificationKWARGS):
    return "Mumbo-"+"-".join(classificationKWARGS["classifiersNames"])

class Mumbo:

    def __init__(self, randomState, NB_CORES=1, **kwargs):
        self.maxIter = kwargs["maxIter"]
        self.minIter = kwargs["minIter"]
        self.threshold = kwargs["threshold"]
        self.classifiersNames = kwargs["classifiersNames"]
        self.classifiersConfigs = kwargs["classifiersConfigs"]
        nbView = kwargs["nbView"]
        self.nbCores = NB_CORES
        self.iterIndex = 0
        self.edges = np.zeros((self.maxIter, nbView))
        self.alphas = np.zeros((self.maxIter, nbView))
        self.generalAlphas = np.zeros(self.maxIter)
        self.bestClassifiers = []
        self.bestViews = np.zeros(self.maxIter, dtype=int)-1
        self.averageAccuracies = np.zeros((self.maxIter, nbView))
        self.iterAccuracies = np.zeros(self.maxIter)
        self.randomState = randomState

    def initDataDependant(self, datasetLength, nbView, nbClass, labels):
        self.edges = np.zeros((self.maxIter, nbView))
        self.alphas = np.zeros((self.maxIter, nbView))
        self.generalAlphas = np.zeros(self.maxIter)
        self.bestClassifiers = []
        self.bestViews = np.zeros(self.maxIter, dtype=int)-1
        self.averageAccuracies = np.zeros((self.maxIter, nbView))
        self.costMatrices = np.array([
                                         np.array([
                                                      np.array([
                                                                   np.array([1 if labels[exampleIndice] != classe
                                                                             else -(nbClass - 1)
                                                                             for classe in range(nbClass)
                                                                             ]) for exampleIndice in range(datasetLength)
                                                                   ]) for viewIndice in range(nbView)])
                                         if iteration == 0
                                         else np.zeros((nbView, datasetLength, nbClass))
                                         for iteration in range(self.maxIter + 1)
                                         ])
        self.generalCostMatrix = np.array([
                                              np.array([
                                                           np.array([1 if labels[exampleIndice] != classe
                                                                     else -(nbClass - 1)
                                                                     for classe in range(nbClass)
                                                                     ]) for exampleIndice in range(datasetLength)
                                                           ]) for iteration in range(self.maxIter)
                                              ])
        self.fs = np.zeros((self.maxIter, nbView, datasetLength, nbClass))
        self.ds = np.zeros((self.maxIter, nbView, datasetLength))
        self.predictions = np.zeros((self.maxIter, nbView, datasetLength))
        self.generalFs = np.zeros((self.maxIter, datasetLength, nbClass))

    def fit_hdf5(self, DATASET, trainIndices=None, viewsIndices=None):

        # Initialization
        if trainIndices is None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if type(viewsIndices)==type(None):
            viewsIndices = range(DATASET.get("Metadata").attrs["nbView"])
        NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
        NB_VIEW = len(viewsIndices)
        DATASET_LENGTH = len(trainIndices)
        LABELS = DATASET.get("Labels").value[trainIndices]
        self.initDataDependant(DATASET_LENGTH, NB_VIEW, NB_CLASS, LABELS)
        # Learning
        isStabilized=False
        self.iterIndex = 0
        while not isStabilized and not self.iterIndex >= self.maxIter-1:
            if self.iterIndex > self.minIter:
                coeffs = np.polyfit(np.log(np.arange(self.iterIndex)+0.00001), self.iterAccuracies[:self.iterIndex], 1)
                if coeffs[0]/self.iterIndex < self.threshold:
                    isStabilized = True

            logging.debug('\t\tStart:\t Iteration ' + str(self.iterIndex + 1))
            classifiers, predictedLabels, areBad = self.trainWeakClassifiers_hdf5(DATASET, trainIndices, NB_CLASS,
                                                                                  DATASET_LENGTH, viewsIndices)
            if areBad.all():
                logging.warning("\t\tWARNING:\tAll bad for iteration " + str(self.iterIndex))

            self.predictions[self.iterIndex] = predictedLabels

            for viewFakeIndex in range(NB_VIEW):
                self.computeEdge(viewFakeIndex, DATASET_LENGTH, LABELS)
                if areBad[viewFakeIndex]:
                    self.alphas[self.iterIndex, viewFakeIndex] = 0.
                else:
                    self.alphas[self.iterIndex, viewFakeIndex] = self.computeAlpha(self.edges[self.iterIndex, viewFakeIndex])

            self.updateDs(LABELS, NB_VIEW, DATASET_LENGTH)
            self.updateFs(NB_VIEW, DATASET_LENGTH, NB_CLASS)

            self.updateCostmatrices(NB_VIEW, DATASET_LENGTH, NB_CLASS, LABELS)
            bestView, edge, bestFakeView = self.chooseView(viewsIndices, LABELS, DATASET_LENGTH)
            self.bestViews[self.iterIndex] = bestView
            logging.debug("\t\t\t Best view : \t\t"+DATASET.get("View"+str(bestView)).attrs["name"])
            if areBad.all():
                self.generalAlphas[self.iterIndex] = 0.
            else:
                self.generalAlphas[self.iterIndex] = self.computeAlpha(edge)
            self.bestClassifiers.append(classifiers[bestFakeView])
            self.updateGeneralFs(DATASET_LENGTH, NB_CLASS, bestFakeView)
            self.updateGeneralCostMatrix(DATASET_LENGTH, NB_CLASS,LABELS)
            predictedLabels = self.predict_hdf5(DATASET, usedIndices=trainIndices, viewsIndices=viewsIndices)
            accuracy = accuracy_score(DATASET.get("Labels").value[trainIndices], predictedLabels)
            self.iterAccuracies[self.iterIndex] = accuracy

            self.iterIndex += 1

    def predict_hdf5(self, DATASET, usedIndices=None, viewsIndices=None):
        NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if viewsIndices is None:
            viewsIndices = range(DATASET.get("Metadata").attrs["nbView"])

        viewDict = dict((viewIndex, index) for index, viewIndex in enumerate(viewsIndices))
        if usedIndices is not None:
            DATASET_LENGTH = len(usedIndices)
            predictedLabels = np.zeros(DATASET_LENGTH)

            for labelIndex, exampleIndex in enumerate(usedIndices):
                votes = np.zeros(NB_CLASS)
                for classifier, alpha, view in zip(self.bestClassifiers, self.alphas, self.bestViews):
                    if view != -1:
                        data = getV(DATASET, int(view), int(exampleIndex))
                        votes[int(classifier.predict(np.array([data])))] += alpha[viewDict[view]]
                    else:
                        pass
                predictedLabels[labelIndex] = np.argmax(votes)
        else:
            predictedLabels = []
        return predictedLabels

    def predict_proba_hdf5(self, DATASET, usedIndices=None):
        NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        DATASET_LENGTH = len(usedIndices)
        predictedProbas = np.zeros((DATASET_LENGTH, NB_CLASS))

        for labelIndex, exampleIndex in enumerate(usedIndices):
            for classifier, alpha, view in zip(self.bestClassifiers, self.alphas, self.bestViews):
                data = getV(DATASET, int(view), exampleIndex)
                predictedProbas[labelIndex, int(classifier.predict(np.array([data])))] += alpha[view]
            predictedProbas[labelIndex,:] = predictedProbas[labelIndex,:]/np.sum(predictedProbas[labelIndex,:])
        return predictedProbas

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

        for viewIndex, (classifier, labelsArray, isBad, averageAccuracy) in enumerate(trainedClassifiersAndLabels):
            self.averageAccuracies[self.iterIndex, viewIndex] = averageAccuracy
            trainedClassifiers.append(classifier)
            labelsMatrix.append(labelsArray)
            areBad.append(isBad)
        return np.array(trainedClassifiers), np.array(labelsMatrix), np.array(areBad)

    def trainWeakClassifiers_hdf5(self, DATASET, trainIndices, NB_CLASS,
                                  DATASET_LENGTH, viewIndices):
        NB_VIEW = len(viewIndices)
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
            delayed(trainWeakClassifier_hdf5)(classifiersNames[classifierIndex],
                                              getV(DATASET,viewIndex,trainIndices),
                                              DATASET.get("Labels").value[trainIndices],
                                              DATASET_LENGTH,
                                              viewIndex, classifiersConfigs[classifierIndex],
                                              DATASET.get("View"+str(viewIndex)).attrs["name"], iterIndex, costMatrices, classifierIndex, self.randomState)
            for classifierIndex, viewIndex in enumerate(viewIndices))

        for viewFakeIndex, (classifier, labelsArray, isBad, averageAccuracy) in enumerate(trainedClassifiersAndLabels):
            self.averageAccuracies[self.iterIndex, viewFakeIndex] = averageAccuracy
            trainedClassifiers.append(classifier)
            labelsMatrix.append(labelsArray)
            areBad.append(isBad)
        return np.array(trainedClassifiers), np.array(labelsMatrix), np.array(areBad)

    def computeEdge(self, viewFakeIndex, DATASET_LENGTH, CLASS_LABELS):
        predictionMatrix = self.predictions[self.iterIndex, viewFakeIndex]
        costMatrix = self.costMatrices[self.iterIndex, viewFakeIndex]
        cCost = float(np.sum(np.array(
            [costMatrix[exampleIndice, int(predictionMatrix[exampleIndice])] for exampleIndice in
             range(DATASET_LENGTH)])))
        tCost = float(np.sum(
            np.array([-costMatrix[exampleIndice, int(CLASS_LABELS[exampleIndice])] for exampleIndice in
                      range(DATASET_LENGTH)])))
        if tCost == 0.:
            self.edges[self.iterIndex, viewFakeIndex] = -cCost
        else:
            self.edges[self.iterIndex, viewFakeIndex] = -cCost / tCost


    def computeAlpha(self, edge):
        if 1 > edge > -1:
            return 0.5 * math.log((1 + edge) / (1 - edge))
        else:
            return 0

    def allViewsClassifyBadly(self, predictions, pastIterIndice, NB_VIEW, CLASS_LABEL, exampleIndice):
        boolean = True
        for viewIndice in range(NB_VIEW):
            if predictions[pastIterIndice, viewIndice, exampleIndice] == CLASS_LABEL:
                boolean = False
        return boolean

    def updateDs(self, CLASS_LABELS, NB_VIEW, DATASET_LENGTH):
        for viewIndice in range(NB_VIEW):
            for exampleIndice in range(DATASET_LENGTH):
                for pastIterIndice in range(self.iterIndex):

                    if self.predictions[pastIterIndice, viewIndice, exampleIndice] \
                            == \
                            CLASS_LABELS[exampleIndice] \
                            or self.allViewsClassifyBadly(self.predictions, pastIterIndice,
                                                         NB_VIEW, CLASS_LABELS[exampleIndice],
                                                         exampleIndice):

                        self.ds[pastIterIndice, viewIndice, exampleIndice] = 1
                    else:
                        self.ds[pastIterIndice, viewIndice, exampleIndice] = 0

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

    def updateCostmatrices(self, NB_VIEW, DATASET_LENGTH, NB_CLASS, CLASS_LABELS):
        for viewIndice in range(NB_VIEW):
            for exampleIndice in range(DATASET_LENGTH):
                for classe in range(NB_CLASS):
                    if classe != CLASS_LABELS[exampleIndice]:
                        self.costMatrices[self.iterIndex + 1, viewIndice, exampleIndice, classe] \
                            = 1.0 * math.exp(self.fs[self.iterIndex, viewIndice, exampleIndice, classe] -
                                             self.fs[self.iterIndex, viewIndice, exampleIndice, int(CLASS_LABELS[exampleIndice])])
                    else:
                        self.costMatrices[self.iterIndex + 1, viewIndice, exampleIndice, classe] \
                            = -1. * np.sum(np.exp(self.fs[self.iterIndex, viewIndice, exampleIndice] -
                                                  self.fs[self.iterIndex, viewIndice, exampleIndice, classe]))
        self.costMatrices /= np.amax(np.absolute(self.costMatrices))

    def chooseView(self, viewIndices, CLASS_LABELS, DATASET_LENGTH):
        for viewIndex in range(len(viewIndices)):
            self.computeEdge(viewIndex, DATASET_LENGTH, CLASS_LABELS)

        bestFakeView = np.argmax(self.edges[self.iterIndex, :])
        bestView = viewIndices[np.argmax(self.edges[self.iterIndex, :])]
        return bestView, self.edges[self.iterIndex, bestFakeView], bestFakeView

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

    def updateGeneralCostMatrix(self, DATASET_LENGTH, NB_CLASS, CLASS_LABELS):
        for exampleIndice in range(DATASET_LENGTH):
            for classe in range(NB_CLASS):
                if classe != CLASS_LABELS[exampleIndice]:
                    self.generalCostMatrix[self.iterIndex, exampleIndice, classe] \
                        = math.exp(self.generalFs[self.iterIndex, exampleIndice, classe] -
                                   self.generalFs[self.iterIndex, exampleIndice, int(CLASS_LABELS[exampleIndice])])
                else:
                    self.generalCostMatrix[self.iterIndex, exampleIndice, classe] \
                        = -1 * np.sum(np.exp(self.generalFs[self.iterIndex, exampleIndice] -
                                             self.generalFs[self.iterIndex, exampleIndice, classe]))

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

    def classifyMumbobyIter_hdf5(self, DATASET, fakeViewsIndicesDict, usedIndices=None, NB_CLASS=2):
        if usedIndices is None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        DATASET_LENGTH = len(usedIndices)
        predictedLabels = np.zeros((DATASET_LENGTH, self.maxIter))
        votes = np.zeros((DATASET_LENGTH, NB_CLASS))

        for iterIndex, (classifier, alpha, view) in enumerate(zip(self.bestClassifiers, self.alphas, self.bestViews)):
            votesByIter = np.zeros((DATASET_LENGTH, NB_CLASS))

            for usedExampleIndex, exampleIndex in enumerate(usedIndices):
                data = np.array([np.array(getV(DATASET,int(view), int(exampleIndex)))])
                votesByIter[usedExampleIndex, int(classifier.predict(data))] += alpha[fakeViewsIndicesDict[view]]
                votes[usedExampleIndex] = votes[usedExampleIndex] + np.array(votesByIter[usedExampleIndex])
                predictedLabels[usedExampleIndex, iterIndex] = np.argmax(votes[usedExampleIndex])

        return np.transpose(predictedLabels)
