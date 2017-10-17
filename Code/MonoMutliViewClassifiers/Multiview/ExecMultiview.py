import sys
import os.path
import errno

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import GetMultiviewDb as DB
import os
import logging
import time
import h5py

import Multiview
from utils.Dataset import getShape
from utils.HyperParameterSearch import searchBestSettings

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def ExecMultiview_multicore(directory, coreIndex, name, learningRate, nbFolds, databaseType, path, LABELS_DICTIONARY,
                            randomState,
                            hyperParamSearch=False, nbCores=1, metrics=None, nIter=30, **arguments):
    DATASET = h5py.File(path + name + str(coreIndex) + ".hdf5", "r")
    return ExecMultiview(directory, DATASET, name, learningRate, nbFolds, 1, databaseType, path, LABELS_DICTIONARY,
                         randomState,
                         hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=nIter, **arguments)


def ExecMultiview(directory, DATASET, name, classificationIndices, KFolds, nbCores, databaseType, path,
                  LABELS_DICTIONARY, randomState,
                  hyperParamSearch=False, metrics=None, nIter=30, **kwargs):
    views = kwargs["views"]
    viewsIndices = kwargs["viewsIndices"]
    if not metrics:
        metrics = [["f1_score", None]]
    CL_type = kwargs["CL_type"]
    classificationKWARGS = kwargs[CL_type + "KWARGS"]
    learningRate = len(classificationIndices[0]) / float(
        (len(classificationIndices[0]) + len(classificationIndices[1])))
    t_start = time.time()
    logging.info("### Main Programm for Multiview Classification")
    logging.info("### Classification - Database : " + str(name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + CL_type + " ; Cores : " + str(nbCores) + ", Train ratio : " + str(learningRate) +
                 ", CV on " + str(KFolds.n_splits) + " folds")

    for viewIndex, viewName in zip(viewsIndices, views):
        logging.info("Info:\t Shape of " + str(viewName) + " :" + str(
            getShape(DATASET, viewIndex)))
    logging.info("Done:\t Read Database Files")

    extractionTime = time.time() - t_start
    learningIndices, validationIndices = classificationIndices
    classifierPackage = getattr(Multiview, CL_type)  # Permet d'appeler un module avec une string
    classifierModule = getattr(classifierPackage, CL_type)
    classifierClass = getattr(classifierModule, CL_type)
    analysisModule = getattr(classifierPackage, "analyzeResults")

    if hyperParamSearch != "None":
        classifier = searchBestSettings(DATASET, CL_type, metrics, learningIndices, KFolds, randomState,
                                        viewsIndices=viewsIndices, searchingTool=hyperParamSearch, nIter=nIter,
                                        **classificationKWARGS)
    else:
        classifier = classifierClass(randomState, NB_CORES=nbCores, **classificationKWARGS)

    classifier.fit_hdf5(DATASET, trainIndices=learningIndices, viewsIndices=viewsIndices, metric=metrics[0])
    trainLabels = classifier.predict_hdf5(DATASET, usedIndices=learningIndices, viewsIndices=viewsIndices)
    testLabels = classifier.predict_hdf5(DATASET, usedIndices=validationIndices, viewsIndices=viewsIndices)
    fullLabels = classifier.predict_hdf5(DATASET, viewsIndices=viewsIndices)
    logging.info("Done:\t Classification")

    classificationTime = time.time() - t_start

    logging.info("Info:\t Time for Classification: " + str(int(classificationTime)) + "[s]")
    logging.info("Start:\t Result Analysis for " + CL_type)

    times = (extractionTime, classificationTime)

    stringAnalysis, imagesAnalysis, metricsScores = analysisModule.execute(classifier, trainLabels,
                                                                           testLabels, DATASET,
                                                                           classificationKWARGS, classificationIndices,
                                                                           LABELS_DICTIONARY, views, nbCores, times,
                                                                           name, KFolds,
                                                                           hyperParamSearch, nIter, metrics,
                                                                           viewsIndices, randomState)
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

    logging.info("Done:\t Result Analysis")
    return CL_type, classificationKWARGS, metricsScores, fullLabels
