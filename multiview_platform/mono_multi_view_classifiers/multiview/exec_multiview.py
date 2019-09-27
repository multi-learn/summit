import errno
import logging
import os
import os.path
import time

import h5py
import numpy as np

from .multiview_utils import MultiviewResult
from . import analyze_results
from .. import multiview_classifiers
from ..utils import hyper_parameter_search
from ..utils.dataset import getShape

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def initConstants(kwargs, classificationIndices, metrics, name, nbCores, KFolds,
                  DATASET):
    """Used to init the constants"""
    views = kwargs["view_names"]
    viewsIndices = kwargs["view_indices"]
    if not metrics:
        metrics = [["f1_score", None]]
    classifier_name = kwargs["classifier_name"]
    classifier_config = kwargs[classifier_name]
    learningRate = len(classificationIndices[0]) / float(
        (len(classificationIndices[0]) + len(classificationIndices[1])))
    t_start = time.time()
    logging.info("Info\t: Classification - Database : " + str(
        name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + classifier_name + " ; Cores : " + str(
        nbCores) + ", Train ratio : " + str(learningRate) +
                 ", CV on " + str(KFolds.n_splits) + " folds")

    for viewIndex, viewName in zip(viewsIndices, views):
        logging.info("Info:\t Shape of " + str(viewName) + " :" + str(
            getShape(DATASET, viewIndex)))
    return classifier_name, t_start, viewsIndices, classifier_config, views, learningRate


def saveResults(classifier, LABELS_DICTIONARY, stringAnalysis, views, classifierModule,
                classificationKWARGS, directory, learningRate, name,
                imagesAnalysis):
    labelsSet = set(LABELS_DICTIONARY.values())
    logging.info(stringAnalysis)
    viewsString = "-".join(views)
    labelsString = "-".join(labelsSet)
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    CL_type_string = classifier.short_name
    outputFileName = directory + "/" + CL_type_string + "/" + timestr + "-results-" + CL_type_string + "-" + viewsString + '-' + labelsString + \
                     '-learnRate_{0:.2f}'.format(learningRate) + '-' + name
    if not os.path.exists(os.path.dirname(outputFileName)):
        try:
            os.makedirs(os.path.dirname(outputFileName))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    outputTextFile = open(outputFileName + '.txt', 'w')
    outputTextFile.write(stringAnalysis)
    outputTextFile.close()

    if imagesAnalysis is not None:
        for imageName in imagesAnalysis.keys():
            if os.path.isfile(outputFileName + imageName + ".png"):
                for i in range(1, 20):
                    testFileName = outputFileName + imageName + "-" + str(
                        i) + ".png"
                    if not os.path.isfile(testFileName):
                        imagesAnalysis[imageName].savefig(testFileName, transparent=True)
                        break

            imagesAnalysis[imageName].savefig(
                outputFileName + imageName + '.png', transparent=True)


def ExecMultiview_multicore(directory, coreIndex, name, learningRate, nbFolds,
                            databaseType, path, LABELS_DICTIONARY,
                            randomState, labels,
                            hyperParamSearch=False, nbCores=1, metrics=None,
                            nIter=30, **arguments):
    """Used to load an HDF5 dataset for each parallel job and execute multiview classification"""
    DATASET = h5py.File(path + name + str(coreIndex) + ".hdf5", "r")
    return ExecMultiview(directory, DATASET, name, learningRate, nbFolds, 1,
                         databaseType, path, LABELS_DICTIONARY,
                         randomState, labels,
                         hyperParamSearch=hyperParamSearch, metrics=metrics,
                         nIter=nIter, **arguments)


def ExecMultiview(directory, DATASET, name, classificationIndices, KFolds,
                  nbCores, databaseType, path,
                  LABELS_DICTIONARY, randomState, labels,
                  hyperParamSearch=False, metrics=None, nIter=30, **kwargs):
    """Used to execute multiview classification and result analysis"""
    logging.debug("Start:\t Initialize constants")
    CL_type, \
    t_start, \
    viewsIndices, \
    classifier_config, \
    views, \
    learningRate = initConstants(kwargs, classificationIndices, metrics, name,
                                 nbCores, KFolds, DATASET)
    logging.debug("Done:\t Initialize constants")

    extractionTime = time.time() - t_start
    logging.info("Info:\t Extraction duration " + str(extractionTime) + "s")

    logging.debug("Start:\t Getting train/test split")
    learningIndices, validationIndices, testIndicesMulticlass = classificationIndices
    logging.debug("Done:\t Getting train/test split")

    logging.debug("Start:\t Getting classifiers modules")
    # classifierPackage = getattr(multiview_classifiers,
    #                             CL_type)  # Permet d'appeler un module avec une string
    classifier_module = getattr(multiview_classifiers, CL_type)
    classifier_name = classifier_module.classifier_class_name
    # classifierClass = getattr(classifierModule, CL_type + "Class")
    logging.debug("Done:\t Getting classifiers modules")

    logging.debug("Start:\t Optimizing hyperparameters")
    if hyperParamSearch != "None":
        classifier_config = hyper_parameter_search.searchBestSettings(DATASET, labels,
                                                               classifier_module,
                                                               classifier_name,
                                                                    metrics[0],
                                                               learningIndices,
                                                               KFolds,
                                                               randomState,
                                                               directory,
                                                               nb_cores=nbCores,
                                                               viewsIndices=viewsIndices,
                                                               searchingTool=hyperParamSearch,
                                                               n_iter=nIter,
                                                               classifier_config=classifier_config)

    classifier = getattr(classifier_module, classifier_name)(randomState,
                                                             **classifier_config)
    logging.debug("Done:\t Optimizing hyperparameters")

    logging.debug("Start:\t Fitting classifier")
    classifier.fit(DATASET, labels, train_indices=learningIndices,
                        view_indices=viewsIndices)
    logging.debug("Done:\t Fitting classifier")

    logging.debug("Start:\t Predicting")
    trainLabels = classifier.predict(DATASET, predict_indices=learningIndices,
                                     view_indices=viewsIndices)
    testLabels = classifier.predict(DATASET, predict_indices=validationIndices,
                                    view_indices=viewsIndices)
    fullLabels = np.zeros(labels.shape, dtype=int) - 100
    for trainIndex, index in enumerate(learningIndices):
        fullLabels[index] = trainLabels[trainIndex]
    for testIndex, index in enumerate(validationIndices):
        fullLabels[index] = testLabels[testIndex]
    if testIndicesMulticlass != []:
        testLabelsMulticlass = classifier.predict_hdf5(DATASET,
                                                       usedIndices=testIndicesMulticlass,
                                                       viewsIndices=viewsIndices)
    else:
        testLabelsMulticlass = []
    logging.info("Done:\t Pertidcting")

    classificationTime = time.time() - t_start
    logging.info("Info:\t Classification duration " + str(extractionTime) + "s")

    # TODO: get better cltype

    logging.info("Start:\t Result Analysis for " + CL_type)
    times = (extractionTime, classificationTime)
    stringAnalysis, imagesAnalysis, metricsScores = analyze_results.execute(
        classifier, trainLabels,
        testLabels, DATASET,
        classifier_config, classificationIndices,
        LABELS_DICTIONARY, views, nbCores, times,
        name, KFolds,
        hyperParamSearch, nIter, metrics,
        viewsIndices, randomState, labels, classifier_module)
    logging.info("Done:\t Result Analysis for " + CL_type)

    logging.debug("Start:\t Saving preds")
    saveResults(classifier, LABELS_DICTIONARY, stringAnalysis, views, classifier_module,
                classifier_config, directory,
                learningRate, name, imagesAnalysis)
    logging.debug("Start:\t Saving preds")

    return MultiviewResult(CL_type, classifier_config, metricsScores,
                           fullLabels, testLabelsMulticlass)
    # return CL_type, classificationKWARGS, metricsScores, fullLabels, testLabelsMulticlass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='This methods is used to execute a multiclass classification with one single view. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('-log', action='store_true',
                               help='Use option to activate Logging to Console')
    groupStandard.add_argument('--type', metavar='STRING', action='store',
                               help='Type of Dataset', default=".hdf5")
    groupStandard.add_argument('--name', metavar='STRING', action='store',
                               help='Name of Database (default: %(default)s)',
                               default='DB')
    groupStandard.add_argument('--view', metavar='STRING', action='store',
                               help='Name of Feature for Classification (default: %(default)s)',
                               default='View0')
    groupStandard.add_argument('--pathF', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)',
                               default='results-FeatExtr/')
    groupStandard.add_argument('--directory', metavar='STRING', action='store',
                               help='Path to the views (default: %(default)s)',
                               default='results-FeatExtr/')
    groupStandard.add_argument('--LABELS_DICTIONARY', metavar='STRING',
                               action='store', nargs='+',
                               help='Name of classLabels CSV-file  (default: %(default)s)',
                               default='classLabels.csv')
    groupStandard.add_argument('--classificationIndices', metavar='STRING',
                               action='store',
                               help='Name of classLabels-Description CSV-file  (default: %(default)s)',
                               default='classLabels-Description.csv')
    groupStandard.add_argument('--nbCores', metavar='INT', action='store',
                               help='Number of cores, -1 for all', type=int,
                               default=1)
    groupStandard.add_argument('--randomState', metavar='INT', action='store',
                               help='Seed for the random state or pickable randomstate file',
                               default=42)
    groupStandard.add_argument('--hyperParamSearch', metavar='STRING',
                               action='store',
                               help='The type of method used tosearch the best set of hyper parameters',
                               default='randomizedSearch')
    groupStandard.add_argument('--metrics', metavar='STRING', action='store',
                               nargs="+",
                               help='metrics used in the experimentation, the first will be the one used in CV',
                               default=[''])
    groupStandard.add_argument('--nIter', metavar='INT', action='store',
                               help='Number of itetarion in hyper parameter search',
                               type=int,
                               default=10)

    args = parser.parse_args()

    directory = args.directory
    name = args.name
    LABELS_DICTIONARY = args.LABELS_DICTIONARY
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

    # Extract the data using MPI ?
    DATASET = None
    labels = None  # (get from CSV ?)

    logfilename = "gen a good logfilename"

    logfile = directory + logfilename
    if os.path.isfile(logfile + ".log"):
        for i in range(1, 20):
            testFileName = logfilename + "-" + str(i) + ".log"
            if not os.path.isfile(directory + testFileName):
                logfile = directory + testFileName
                break
    else:
        logfile += ".log"

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        filename=logfile, level=logging.DEBUG,
                        filemode='w')

    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())

    res = ExecMultiview(directory, DATASET, name, classificationIndices, KFolds,
                        nbCores, databaseType, path,
                        LABELS_DICTIONARY, randomState, labels,
                        hyperParamSearch=hyperParamSearch, metrics=metrics,
                        nIter=nIter, **kwargs)

    # Pickle the res
    # Go put your token
