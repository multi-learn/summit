#!/usr/bin/env python

""" Execution: Script to perform a MonoView classification """

import errno
import logging  # To create Log-Files
# Import built-in modules
import os  # to geth path of the running script
import time  # for time calculations

import h5py
# Import 3rd party modules
import numpy as np  # for reading CSV-files and Series

from . import monoview_utils
from .analyze_result import execute
# Import own modules
from .. import monoview_classifiers
from ..utils.dataset import getValue, extractSubset
from ..utils import hyper_parameter_search

# Author-Info
__author__ = "Nikolas Huelsmann, Baptiste BAUVIN"
__status__ = "Prototype"  # Production, Development, Prototype


# __date__ = 2016 - 03 - 25


def ExecMonoview_multicore(directory, name, labelsNames, classificationIndices,
                           KFolds, datasetFileIndex, databaseType,
                           path, randomState, labels,
                           hyperParamSearch="randomizedSearch",
                           metrics=[["accuracy_score", None]], nIter=30,
                           **args):
    DATASET = h5py.File(path + name + str(datasetFileIndex) + ".hdf5", "r")
    neededViewIndex = args["viewIndex"]
    X = DATASET.get("View" + str(neededViewIndex))
    Y = labels
    return ExecMonoview(directory, X, Y, name, labelsNames,
                        classificationIndices, KFolds, 1, databaseType, path,
                        randomState, hyperParamSearch=hyperParamSearch,
                        metrics=metrics, nIter=nIter, **args)


def ExecMonoview(directory, X, Y, name, labelsNames, classificationIndices,
                 KFolds, nbCores, databaseType, path,
                 randomState, hyperParamSearch="randomizedSearch",
                 metrics=[["accuracy_score", None]], nIter=30, **args):
    logging.debug("Start:\t Loading data")
    kwargs, \
    t_start, \
    feat, \
    CL_type, \
    X, \
    learningRate, \
    labelsString, \
    outputFileName = initConstants(args, X, classificationIndices, labelsNames,
                                   name, directory)
    logging.debug("Done:\t Loading data")

    logging.debug(
        "Info:\t Classification - Database:" + str(name) + " Feature:" + str(
            feat) + " train ratio:"
        + str(learningRate) + ", CrossValidation k-folds: " + str(
            KFolds.n_splits) + ", cores:"
        + str(nbCores) + ", algorithm : " + CL_type)

    logging.debug("Start:\t Determine Train/Test split")
    X_train, y_train, X_test, y_test, X_test_multiclass = initTrainTest(X, Y,
                                                                        classificationIndices)

    logging.debug("Info:\t Shape X_train:" + str(
        X_train.shape) + ", Length of y_train:" + str(len(y_train)))
    logging.debug("Info:\t Shape X_test:" + str(
        X_test.shape) + ", Length of y_test:" + str(len(y_test)))
    logging.debug("Done:\t Determine Train/Test split")

    logging.debug("Start:\t Generate classifier args")
    classifierModuleName = CL_type.split("_")[0]
    classifierModule = getattr(monoview_classifiers, classifierModuleName)
    clKWARGS, testFoldsPreds = getHPs(classifierModule, hyperParamSearch,
                                      nIter, CL_type, X_train, y_train,
                                      randomState, outputFileName,
                                      KFolds, nbCores, metrics, kwargs)
    logging.debug("Done:\t Generate classifier args")

    logging.debug("Start:\t Training")
    classifier = getattr(classifierModule, classifierModuleName)(randomState, **clKWARGS)

    classifier.fit(X_train, y_train)  # NB_CORES=nbCores,
    logging.debug("Done:\t Training")

    logging.debug("Start:\t Predicting")
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)
    full_labels_pred = np.zeros(Y.shape, dtype=int) - 100
    for trainIndex, index in enumerate(classificationIndices[0]):
        full_labels_pred[index] = y_train_pred[trainIndex]
    for testIndex, index in enumerate(classificationIndices[1]):
        full_labels_pred[index] = y_test_pred[testIndex]
    if X_test_multiclass != []:
        y_test_multiclass_pred = classifier.predict(X_test_multiclass)
    else:
        y_test_multiclass_pred = []
    logging.debug("Done:\t Predicting")

    t_end = time.time() - t_start
    logging.debug(
        "Info:\t Time for training and predicting: " + str(t_end) + "[s]")

    logging.debug("Start:\t Getting results")
    stringAnalysis, \
    imagesAnalysis, \
    metricsScores = execute(name, classificationIndices, KFolds, nbCores,
                            hyperParamSearch, metrics, nIter, feat, CL_type,
                            clKWARGS, labelsNames, X.shape,
                            y_train, y_train_pred, y_test, y_test_pred, t_end,
                            randomState, classifier, outputFileName)
    # cl_desc = [value for key, value in sorted(clKWARGS.items())]
    logging.debug("Done:\t Getting results")

    logging.debug("Start:\t Saving preds")
    saveResults(stringAnalysis, outputFileName, full_labels_pred, y_train_pred,
                y_train, imagesAnalysis, y_test)
    logging.info("Done:\t Saving results")

    viewIndex = args["viewIndex"]
    if testFoldsPreds is None:
        testFoldsPreds = y_train_pred
    return monoview_utils.MonoviewResult(viewIndex, CL_type, feat, metricsScores,
                                         full_labels_pred, clKWARGS,
                                         y_test_multiclass_pred, testFoldsPreds)
    # return viewIndex, [CL_type, feat, metricsScores, full_labels_pred, clKWARGS, y_test_multiclass_pred, testFoldsPreds]


def initConstants(args, X, classificationIndices, labelsNames, name, directory):
    try:
        kwargs = args["args"]
    except KeyError:
        kwargs = args
    t_start = time.time()
    if type(X.attrs["name"]) == bytes:
        feat = X.attrs["name"].decode("utf-8")
    else:
        feat = X.attrs["name"]
    CL_type = kwargs["CL_type"]
    X = getValue(X)
    learningRate = float(len(classificationIndices[0])) / (
                len(classificationIndices[0]) + len(classificationIndices[1]))
    labelsString = "-".join(labelsNames)
    CL_type_string = CL_type
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    outputFileName = directory + CL_type_string + "/" + feat + "/" + timestr + "-results-" + CL_type_string + "-" + labelsString + \
                     '-learnRate_{0:.2f}'.format(
                         learningRate) + '-' + name + "-" + feat + "-"
    if not os.path.exists(os.path.dirname(outputFileName)):
        try:
            os.makedirs(os.path.dirname(outputFileName))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return kwargs, t_start, feat, CL_type, X, learningRate, labelsString, outputFileName


def initTrainTest(X, Y, classificationIndices):
    trainIndices, testIndices, testIndicesMulticlass = classificationIndices
    X_train = extractSubset(X, trainIndices)
    X_test = extractSubset(X, testIndices)
    if np.array(testIndicesMulticlass).size != 0:
        X_test_multiclass = extractSubset(X, testIndicesMulticlass)
    else:
        X_test_multiclass = []
    y_train = Y[trainIndices]
    y_test = Y[testIndices]
    return X_train, y_train, X_test, y_test, X_test_multiclass


def getHPs(classifierModule, hyperParamSearch, nIter, CL_type, X_train, y_train,
           randomState,
           outputFileName, KFolds, nbCores, metrics, kwargs):
    if hyperParamSearch != "None":
        logging.debug(
            "Start:\t " + hyperParamSearch + " best settings with " + str(
                nIter) + " iterations for " + CL_type)
        classifierHPSearch = getattr(hyper_parameter_search, hyperParamSearch)
        clKWARGS, testFoldsPreds = classifierHPSearch(X_train, y_train, "monoview",
                                                      randomState,
                                                      outputFileName,
                                                      classifierModule, CL_type,
                                                      folds=KFolds,
                                                      nb_cores=nbCores,
                                                      metric=metrics[0],
                                                      n_iter=nIter,
                                                      classifier_kwargs=kwargs[
                                                          CL_type + "KWARGS"])
        logging.debug("Done:\t " + hyperParamSearch + " best settings")
    else:
        clKWARGS = kwargs[CL_type + "KWARGS"]
        testFoldsPreds = None
    return clKWARGS, testFoldsPreds


def saveResults(stringAnalysis, outputFileName, full_labels_pred, y_train_pred,
                y_train, imagesAnalysis, y_test):
    logging.info(stringAnalysis)
    outputTextFile = open(outputFileName + 'summary.txt', 'w')
    outputTextFile.write(stringAnalysis)
    outputTextFile.close()
    np.savetxt(outputFileName + "full_pred.csv",
               full_labels_pred.astype(np.int16), delimiter=",")
    np.savetxt(outputFileName + "train_pred.csv", y_train_pred.astype(np.int16),
               delimiter=",")
    np.savetxt(outputFileName + "train_labels.csv", y_train.astype(np.int16),
               delimiter=",")
    np.savetxt(outputFileName + "test_labels.csv", y_test.astype(np.int16),
               delimiter=",")

    if imagesAnalysis is not None:
        for imageName in imagesAnalysis:
            if os.path.isfile(outputFileName + imageName + ".png"):
                for i in range(1, 20):
                    testFileName = outputFileName + imageName + "-" + str(
                        i) + ".png"
                    if not os.path.isfile(testFileName):
                        imagesAnalysis[imageName].savefig(testFileName, transparent=True)
                        break

            imagesAnalysis[imageName].savefig(
                outputFileName + imageName + '.png', transparent=True)


# if __name__ == '__main__':
#     """The goal of this part of the module is to be able to execute a monoview experimentation
#      on a node of a cluster independently.
#      So one need to fill in all the ExecMonoview function arguments with the parse arg function
#      It could be a good idea to use pickle to store all the 'simple' args in order to reload them easily"""
#     import argparse
#     import pickle
#
#     from ..utils import dataset
#
#     parser = argparse.ArgumentParser(
#         description='This methods is used to execute a multiclass classification with one single view. ',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
#     groupStandard = parser.add_argument_group('Standard arguments')
#     groupStandard.add_argument('-log', action='store_true',
#                                help='Use option to activate Logging to Console')
#     groupStandard.add_argument('--name', metavar='STRING', action='store',
#                                help='Name of Database', default='Plausible')
#     groupStandard.add_argument('--cl_name', metavar='STRING', action='store',
#                                help='THe name of the monoview classifier to use',
#                                default='DecisionTree')
#     groupStandard.add_argument('--view', metavar='STRING', action='store',
#                                help='Name of the view used', default='View0')
#     groupStandard.add_argument('--pathF', metavar='STRING', action='store',
#                                help='Path to the database hdf5 file',
#                                default='../../../Data/Plausible')
#     groupStandard.add_argument('--directory', metavar='STRING', action='store',
#                                help='Path of the output directory', default='')
#     groupStandard.add_argument('--labelsNames', metavar='STRING',
#                                action='store', nargs='+',
#                                help='Name of the labels used for classification',
#                                default=['Yes', 'No'])
#     groupStandard.add_argument('--classificationIndices', metavar='STRING',
#                                action='store',
#                                help='Path to the classificationIndices pickle file',
#                                default='')
#     groupStandard.add_argument('--KFolds', metavar='STRING', action='store',
#                                help='Path to the kFolds pickle file',
#                                default='')
#     groupStandard.add_argument('--nbCores', metavar='INT', action='store',
#                                help='Number of cores, -1 for all',
#                                type=int, default=1)
#     groupStandard.add_argument('--randomState', metavar='INT', action='store',
#                                help='Seed for the random state or pickable randomstate file',
#                                default=42)
#     groupStandard.add_argument('--hyperParamSearch', metavar='STRING',
#                                action='store',
#                                help='The type of method used to search the best set of hyper parameters',
#                                default='randomizedSearch')
#     groupStandard.add_argument('--metrics', metavar='STRING', action='store',
#                                help='Path to the pickle file describing the metricsused to analyze the performance',
#                                default='')
#     groupStandard.add_argument('--kwargs', metavar='STRING', action='store',
#                                help='Path to the pickle file containing the key-words arguments used for classification',
#                                default='')
#     groupStandard.add_argument('--nIter', metavar='INT', action='store',
#                                help='Number of itetarion in hyper parameter search',
#                                type=int,
#                                default=10)
#
#     args = parser.parse_args()
#
#     directory = args.directory
#     name = args.name
#     classifierName = args.cl_name
#     labelsNames = args.labelsNames
#     viewName = args.view
#     with open(args.classificationIndices, 'rb') as handle:
#         classificationIndices = pickle.load(handle)
#     with open(args.KFolds, 'rb') as handle:
#         KFolds = pickle.load(handle)
#     nbCores = args.nbCores
#     path = args.pathF
#     with open(args.randomState, 'rb') as handle:
#         randomState = pickle.load(handle)
#     hyperParamSearch = args.hyperParamSearch
#     with open(args.metrics, 'rb') as handle:
#         metrics = pickle.load(handle)
#     nIter = args.nIter
#     with open(args.kwargs, 'rb') as handle:
#         kwargs = pickle.load(handle)
#
#     databaseType = None
#
#     # Extract the data using MPI
#     X, Y = dataset.getMonoviewShared(path, name, viewName)
#
#     # Init log
#     logFileName = time.strftime(
#         "%Y_%m_%d-%H_%M_%S") + "-" + name + "-" + viewName + "-" + classifierName + '-LOG'
#     if not os.path.exists(os.path.dirname(directory + logFileName)):
#         try:
#             os.makedirs(os.path.dirname(directory + logFileName))
#         except OSError as exc:
#             if exc.errno != errno.EEXIST:
#                 raise
#     logFile = directory + logFileName
#     if os.path.isfile(logFile + ".log"):
#         for i in range(1, 20):
#             testFileName = logFileName + "-" + str(i) + ".log"
#             if not (os.path.isfile(directory + testFileName)):
#                 logFile = directory + testFileName
#                 break
#     else:
#         logFile += ".log"
#     logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
#                         filename=logFile, level=logging.DEBUG,
#                         filemode='w')
#     if args.log:
#         logging.getLogger().addHandler(logging.StreamHandler())
#
#     # Computing on multiple cores
#     res = ExecMonoview(directory, X, Y, name, labelsNames,
#                        classificationIndices, KFolds, nbCores, databaseType,
#                        path,
#                        randomState, hyperParamSearch=hyperParamSearch,
#                        metrics=metrics, nIter=nIter, **kwargs)
#
#     with open(directory + "res.pickle", "wb") as handle:
#         pickle.dump(res, handle)

    # Pickle the res in a file to be reused.
    # Go put a token in the token files without breaking everything.

    # Need to write a function to be  able to know the timeu sed
    # for a monoview experimentation approximately and the ressource it uses to write automatically the file in the shell
    # it will have to be a not-too close approx as the taskswont be long and Ram-o-phage
