#!/usr/bin/env python

""" Execution: Script to perform a MonoView classification """

# Import built-in modules
import os  # to geth path of the running script
import time  # for time calculations
import errno

# Import 3rd party modules
import numpy as np  # for reading CSV-files and Series
import logging  # To create Log-Files
import h5py

# Import own modules
from .. import MonoviewClassifiers
from .analyzeResult import execute
from ..utils.Dataset import getValue, extractSubset

# Author-Info
__author__ = "Nikolas Huelsmann, Baptiste BAUVIN"
__status__ = "Prototype"  # Production, Development, Prototype
# __date__ = 2016 - 03 - 25


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
    learningRate = float(len(classificationIndices[0])) / (len(classificationIndices[0]) + len(classificationIndices[1]))
    labelsString = "-".join(labelsNames)
    CL_type_string = CL_type

    outputFileName = directory + CL_type_string + "/" + feat + "/" + "Results-" + CL_type_string + "-" + labelsString + \
                     '-learnRate' + str(learningRate) + '-' + name + "-" + feat + "-"
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
    if testIndicesMulticlass != []:
        X_test_multiclass = extractSubset(X, testIndicesMulticlass)
    else:
        X_test_multiclass = []
    y_train = Y[trainIndices]
    y_test = Y[testIndices]
    return X_train, y_train, X_test, y_test, X_test_multiclass


def getKWARGS(classifierModule, hyperParamSearch, nIter, CL_type, X_train, y_train, randomState,
              outputFileName, KFolds, nbCores, metrics, kwargs):
    if hyperParamSearch != "None":
        logging.debug("Start:\t " + hyperParamSearch + " best settings with " + str(nIter) + " iterations for " + CL_type)
        classifierHPSearch = getattr(classifierModule, hyperParamSearch)
        cl_desc = classifierHPSearch(X_train, y_train, randomState, outputFileName, KFolds=KFolds, nbCores=nbCores,
                                     metric=metrics[0], nIter=nIter)
        clKWARGS = dict((str(index), desc) for index, desc in enumerate(cl_desc))
        logging.debug("Done:\t " + hyperParamSearch + "RandomSearch best settings")
    else:
        clKWARGS = kwargs[CL_type + "KWARGS"]
    return clKWARGS


def saveResults(stringAnalysis, outputFileName, full_labels_pred, y_train_pred, y_train, imagesAnalysis):
    logging.info(stringAnalysis)
    outputTextFile = open(outputFileName + '.txt', 'w')
    outputTextFile.write(stringAnalysis)
    outputTextFile.close()
    np.savetxt(outputFileName + "full_pred.csv", full_labels_pred.astype(np.int16), delimiter=",")
    np.savetxt(outputFileName + "train_pred.csv", y_train_pred.astype(np.int16), delimiter=",")
    np.savetxt(outputFileName + "train_labels.csv", y_train.astype(np.int16), delimiter=",")

    if imagesAnalysis is not None:
        for imageName in imagesAnalysis:
            if os.path.isfile(outputFileName + imageName + ".png"):
                for i in range(1, 20):
                    testFileName = outputFileName + imageName + "-" + str(i) + ".png"
                    if not os.path.isfile(testFileName):
                        imagesAnalysis[imageName].savefig(testFileName)
                        break

            imagesAnalysis[imageName].savefig(outputFileName + imageName + '.png')


def ExecMonoview_multicore(directory, name, labelsNames, classificationIndices, KFolds, datasetFileIndex, databaseType,
                           path, randomState, labels, hyperParamSearch="randomizedSearch",
                           metrics=[["accuracy_score", None]], nIter=30, **args):
    DATASET = h5py.File(path + name + str(datasetFileIndex) + ".hdf5", "r")
    # kwargs = args["args"]
    # views = [DATASET.get("View" + str(viewIndex)).attrs["name"] for viewIndex in
    #          range(DATASET.get("Metadata").attrs["nbView"])]
    neededViewIndex = args["viewIndex"]
    X = DATASET.get("View" + str(neededViewIndex))
    Y = labels
    return ExecMonoview(directory, X, Y, name, labelsNames, classificationIndices, KFolds, 1, databaseType, path,
                        randomState, hyperParamSearch=hyperParamSearch,
                        metrics=metrics, nIter=nIter, **args)


def ExecMonoview(directory, X, Y, name, labelsNames, classificationIndices, KFolds, nbCores, databaseType, path,
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
    outputFileName = initConstants(args, X, classificationIndices, labelsNames, name, directory)
    logging.debug("Done:\t Loading data")

    logging.debug("Info:\t Classification - Database:" + str(name) + " Feature:" + str(feat) + " train ratio:"
                  + str(learningRate) + ", CrossValidation k-folds: " + str(KFolds.n_splits) + ", cores:"
                  + str(nbCores) + ", algorithm : " + CL_type)

    logging.debug("Start:\t Determine Train/Test split")
    X_train, y_train, X_test, y_test, X_test_multiclass = initTrainTest(X, Y, classificationIndices)
    logging.debug("Info:\t Shape X_train:" + str(X_train.shape) + ", Length of y_train:" + str(len(y_train)))
    logging.debug("Info:\t Shape X_test:" + str(X_test.shape) + ", Length of y_test:" + str(len(y_test)))
    logging.debug("Done:\t Determine Train/Test split")

    logging.debug("Start:\t Generate classifier args")
    classifierModule = getattr(MonoviewClassifiers, CL_type)
    clKWARGS = getKWARGS(classifierModule, hyperParamSearch,
                         nIter, CL_type, X_train, y_train,
                         randomState, outputFileName,
                         KFolds, nbCores, metrics, kwargs)
    logging.debug("Done:\t Generate classifier args")

    logging.debug("Start:\t Training")
    cl_res = classifierModule.fit(X_train, y_train, randomState, NB_CORES=nbCores, **clKWARGS)
    logging.debug("Done:\t Training")

    logging.debug("Start:\t Predicting")
    y_train_pred = cl_res.predict(X_train)
    y_test_pred = cl_res.predict(X_test)
    full_labels_pred = np.zeros(Y.shape, dtype=int)-100
    for trainIndex, index in enumerate(classificationIndices[0]):
        full_labels_pred[index] = y_train_pred[trainIndex]
    for testIndex, index in enumerate(classificationIndices[1]):
        full_labels_pred[index] = y_test_pred[testIndex]
    if X_test_multiclass != []:
        y_test_multiclass_pred = cl_res.predict(X_test_multiclass)
    else:
        y_test_multiclass_pred = []
    logging.debug("Done:\t Predicting")

    t_end = time.time() - t_start
    logging.debug("Info:\t Time for training and predicting: " + str(t_end) + "[s]")

    logging.debug("Start:\t Getting Results")
    stringAnalysis, \
    imagesAnalysis, \
    metricsScores = execute(name, classificationIndices, KFolds, nbCores,
                                                            hyperParamSearch, metrics, nIter, feat, CL_type,
                                                            clKWARGS, labelsNames, X.shape,
                                                            y_train, y_train_pred, y_test, y_test_pred, t_end,
                                                            randomState, cl_res, outputFileName)
    cl_desc = [value for key, value in sorted(clKWARGS.items())]
    logging.debug("Done:\t Getting Results")

    logging.debug("Start:\t Saving preds")
    saveResults(stringAnalysis, outputFileName, full_labels_pred, y_train_pred, y_train, imagesAnalysis)
    logging.info("Done:\t Saving Results")

    viewIndex = args["viewIndex"]
    return viewIndex, [CL_type, cl_desc + [feat], metricsScores, full_labels_pred, clKWARGS, y_test_multiclass_pred]


if __name__ == '__main__':
    """The goal of this part of the module is to be able to execute a monoview experimentation
     on a node of a cluster independently.
     So one need to fill in all the ExecMonoview function arguments with the parse arg function
     It could be a good idea to use pickle to store all the 'simple' args in order to reload them easily"""
    import argparse

    parser = argparse.ArgumentParser(
        description='This methods is used to execute a multiclass classification with one single view. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
    groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of Dataset', default=".hdf5")
    groupStandard.add_argument('--name', metavar='STRING', action='store',
                               help='Name of Database (default: %(default)s)', default='DB')
    groupStandard.add_argument('--view', metavar='STRING', action='store',
                               help='Name of Feature for Classification (default: %(default)s)', default='View0')
    groupStandard.add_argument('--pathF', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)', default='Results-FeatExtr/')
    groupStandard.add_argument('--directory', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)', default='Results-FeatExtr/')
    groupStandard.add_argument('--labelsNames', metavar='STRING', action='store', nargs='+',
                               help='Name of classLabels CSV-file  (default: %(default)s)', default='classLabels.csv')
    groupStandard.add_argument('--classificationIndices', metavar='STRING', action='store',
                               help='Name of classLabels-Description CSV-file  (default: %(default)s)',
                               default='classLabels-Description.csv')
    groupStandard.add_argument('--nbCores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                            default=1)
    groupStandard.add_argument('--randomState', metavar='INT', action='store',
                               help='Seed for the random state or pickable randomstate file', default=42)
    groupStandard.add_argument('--hyperParamSearch', metavar='STRING', action='store',
                               help='The type of method used tosearch the best set of hyper parameters', default='randomizedSearch')
    groupStandard.add_argument('--metrics', metavar='STRING', action='store', nargs="+",
                               help='Metrics used in the experimentation, the first will be the one used in CV',
                               default=[''])
    groupStandard.add_argument('--nIter', metavar='INT', action='store', help='Number of itetarion in hyper parameter search', type=int,
                               default=10)

    args = parser.parse_args()

    directory = args.directory
    name = args.name
    labelsNames = args.labelsNames
    classificationIndices = args.classificationIndices
    KFolds = args.KFolds
    nbCores = args.nbCores
    databaseType = None
    path = args.pathF
    randomState = args.randomState
    hyperParamSearch = args.hyperParamSearch
    metrics = args.metrics
    nIter = args.nIter
    kwargs = args.kwargs

    # Extract the data using MPI
    X = None
    Y = None

    logfilename = "gen a goodlogfilename"



    logfile = directory + logfilename
    if os.path.isfile(logfile + ".log"):
        for i in range(1, 20):
            testFileName = logfilename + "-" + str(i) + ".log"
            if not os.path.isfile(directory + testFileName):
                logfile = directory + testFileName
                break
    else:
        logfile += ".log"

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logfile, level=logging.DEBUG,
                        filemode='w')

    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())


    res = ExecMonoview(directory, X, Y, name, labelsNames, classificationIndices, KFolds, nbCores, databaseType, path,
                 randomState, hyperParamSearch=hyperParamSearch,
                 metrics=metrics, nIter=nIter, **kwargs)


    # Pickle the res in a file to be reused.
    # Go put a token in the token files without breaking everything.

    # Need to write a function to be  able to know the timeu sed
    # for a monoview experimentation approximately and the ressource it uses to write automatically the file in the shell
    # it will have to be a not-too close approx as the taskswont be long and Ram-o-phage