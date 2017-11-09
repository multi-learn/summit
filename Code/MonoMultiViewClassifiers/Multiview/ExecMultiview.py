import errno
import logging
import os
import os.path
import time
import h5py
import numpy as np

from ..utils import HyperParameterSearch
from ..utils.Dataset import getShape
from .. import MultiviewClassifiers


# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def initConstants(kwargs, classificationIndices, metrics, name, nbCores, KFolds, DATASET):
    """Used to init the constants"""
    views = kwargs["views"]
    viewsIndices = kwargs["viewsIndices"]
    if not metrics:
        metrics = [["f1_score", None]]
    CL_type = kwargs["CL_type"]
    classificationKWARGS = kwargs[CL_type + "KWARGS"]
    learningRate = len(classificationIndices[0]) / float(
        (len(classificationIndices[0]) + len(classificationIndices[1])))
    t_start = time.time()
    logging.info("Info\t: Classification - Database : " + str(name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + CL_type + " ; Cores : " + str(nbCores) + ", Train ratio : " + str(learningRate) +
                 ", CV on " + str(KFolds.n_splits) + " folds")

    for viewIndex, viewName in zip(viewsIndices, views):
        logging.info("Info:\t Shape of " + str(viewName) + " :" + str(
            getShape(DATASET, viewIndex)))
    return CL_type, t_start, viewsIndices, classificationKWARGS, views, learningRate


def saveResults(LABELS_DICTIONARY, stringAnalysis, views, classifierModule, classificationKWARGS, directory, learningRate, name, imagesAnalysis):
    labelsSet = set(LABELS_DICTIONARY.values())
    logging.info(stringAnalysis)
    featureString = "-".join(views)
    labelsString = "-".join(labelsSet)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    CL_type_string = classifierModule.getCLString(classificationKWARGS)
    outputFileName = directory + "/" + CL_type_string + "/" + timestr + "Results-" + CL_type_string + "-" + featureString + '-' + labelsString + \
                     '-learnRate' + str(learningRate) + '-' + name
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
        for imageName in imagesAnalysis:
            if os.path.isfile(outputFileName + imageName + ".png"):
                for i in range(1, 20):
                    testFileName = outputFileName + imageName + "-" + str(i) + ".png"
                    if not os.path.isfile(testFileName):
                        imagesAnalysis[imageName].savefig(testFileName)
                        break

            imagesAnalysis[imageName].savefig(outputFileName + imageName + '.png')


def ExecMultiview_multicore(directory, coreIndex, name, learningRate, nbFolds, databaseType, path, LABELS_DICTIONARY,
                            randomState, labels,
                            hyperParamSearch=False, nbCores=1, metrics=None, nIter=30, **arguments):
    """Used to load an HDF5 dataset for each parallel job and execute multiview classification"""
    DATASET = h5py.File(path + name + str(coreIndex) + ".hdf5", "r")
    return ExecMultiview(directory, DATASET, name, learningRate, nbFolds, 1, databaseType, path, LABELS_DICTIONARY,
                         randomState, labels,
                         hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=nIter, **arguments)


def ExecMultiview(directory, DATASET, name, classificationIndices, KFolds, nbCores, databaseType, path,
                  LABELS_DICTIONARY, randomState, labels,
                  hyperParamSearch=False, metrics=None, nIter=30, **kwargs):
    """Used to execute multiview classification and result analysis"""
    logging.debug("Start:\t Initialize constants")
    CL_type, \
    t_start, \
    viewsIndices, \
    classificationKWARGS, \
    views, \
    learningRate = initConstants(kwargs, classificationIndices, metrics,name, nbCores,KFolds, DATASET)
    logging.debug("Done:\t Initialize constants")

    extractionTime = time.time() - t_start
    logging.info("Info:\t Extraction duration "+str(extractionTime)+"s")

    logging.debug("Start:\t Getting train/test split")
    learningIndices, validationIndices, testIndicesMulticlass = classificationIndices
    logging.debug("Done:\t Getting train/test split")

    logging.debug("Start:\t Getting classifiers modules")
    classifierPackage = getattr(MultiviewClassifiers, CL_type)  # Permet d'appeler un module avec une string
    classifierModule = getattr(classifierPackage, CL_type+"Module")
    classifierClass = getattr(classifierModule, CL_type+"Class")
    analysisModule = getattr(classifierPackage, "analyzeResults")
    logging.debug("Done:\t Getting classifiers modules")

    logging.debug("Start:\t Optimizing hyperparameters")
    if hyperParamSearch != "None":
        classifier = HyperParameterSearch.searchBestSettings(DATASET, labels, classifierPackage,
                                                             CL_type, metrics, learningIndices,
                                                             KFolds, randomState,
                                                             viewsIndices=viewsIndices,
                                                             searchingTool=hyperParamSearch, nIter=nIter,
                                                             **classificationKWARGS)
    else:
        classifier = classifierClass(randomState, NB_CORES=nbCores, **classificationKWARGS)
    logging.debug("Done:\t Optimizing hyperparameters")

    logging.debug("Start:\t Fitting classifier")
    classifier.fit_hdf5(DATASET, labels, trainIndices=learningIndices, viewsIndices=viewsIndices, metric=metrics[0])
    logging.debug("Done:\t Fitting classifier")

    logging.debug("Start:\t Predicting")
    trainLabels = classifier.predict_hdf5(DATASET, usedIndices=learningIndices, viewsIndices=viewsIndices)
    testLabels = classifier.predict_hdf5(DATASET, usedIndices=validationIndices, viewsIndices=viewsIndices)
    fullLabels = np.zeros(labels.shape, dtype=int)-100
    for trainIndex, index in enumerate(learningIndices):
        fullLabels[index] = trainLabels[trainIndex]
    for testIndex, index in enumerate(validationIndices):
        fullLabels[index] = testLabels[testIndex]
    if testIndicesMulticlass:
        testLabelsMulticlass = classifier.predict_hdf5(DATASET, usedIndices=testIndicesMulticlass, viewsIndices=viewsIndices)
    else:
        testLabelsMulticlass = []
    logging.info("Done:\t Pertidcting")

    classificationTime = time.time() - t_start
    logging.info("Info:\t Classification duration " + str(extractionTime) + "s")


    #TODO: get better cltype

    logging.info("Start:\t Result Analysis for " + CL_type)
    times = (extractionTime, classificationTime)
    stringAnalysis, imagesAnalysis, metricsScores = analysisModule.execute(classifier, trainLabels,
                                                                           testLabels, DATASET,
                                                                           classificationKWARGS, classificationIndices,
                                                                           LABELS_DICTIONARY, views, nbCores, times,
                                                                           name, KFolds,
                                                                           hyperParamSearch, nIter, metrics,
                                                                           viewsIndices, randomState, labels)
    logging.info("Done:\t Result Analysis for " + CL_type)

    logging.debug("Start:\t Saving preds")
    saveResults(LABELS_DICTIONARY, stringAnalysis, views, classifierModule, classificationKWARGS, directory,
                learningRate, name, imagesAnalysis)
    logging.debug("Start:\t Saving preds")

    return CL_type, classificationKWARGS, metricsScores, fullLabels, testLabelsMulticlass
