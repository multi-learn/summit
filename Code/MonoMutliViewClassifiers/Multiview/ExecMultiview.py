import sys
import os.path

sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from Multiview import *

import GetMultiviewDb as DB
import argparse
import os
import logging
import time
import h5py
from utils.Dataset import getShape
from utils.HyperParameterSearch import searchBestSettings

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"                           # Production, Development, Prototype


def ExecMultiview_multicore(directory, coreIndex, name, learningRate, nbFolds, databaseType, path, LABELS_DICTIONARY, statsIter,
                            hyperParamSearch=False, nbCores=1, metrics=None, nIter=30, **arguments):
    DATASET = h5py.File(path+name+str(coreIndex)+".hdf5", "r")
    return ExecMultiview(directory, DATASET, name, learningRate, nbFolds, 1, databaseType, path, LABELS_DICTIONARY, statsIter,
                         hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=nIter, **arguments)


def ExecMultiview(directory, DATASET, name, learningRate, nbFolds, nbCores, databaseType, path, LABELS_DICTIONARY, statsIter,
                  hyperParamSearch=False, metrics=None, nIter=30, **kwargs):

    datasetLength = DATASET.get("Metadata").attrs["datasetLength"]
    NB_VIEW = kwargs["NB_VIEW"]
    views = kwargs["views"]
    viewsIndices = kwargs["viewsIndices"]
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]
    if not metrics:
        metrics = [["accuracy_score", None]]
    metric = metrics[0]
    CL_type = kwargs["CL_type"]
    LABELS_NAMES = kwargs["LABELS_NAMES"]
    classificationKWARGS = kwargs[CL_type+"KWARGS"]

    t_start = time.time()
    logging.info("### Main Programm for Multiview Classification")
    logging.info("### Classification - Database : " + str(name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + CL_type + " ; Cores : " + str(nbCores))

    for viewIndex, viewName in zip(viewsIndices, views):
        logging.info("Info:\t Shape of " + str(viewName) + " :" + str(
                getShape(DATASET, viewIndex)))
    logging.info("Done:\t Read Database Files")

    extractionTime = time.time() - t_start
    ivalidationIndices = []
    trainLabelsIterations = []
    testLabelsIterations = []
    classifiersIterations = []
    classifierPackage = globals()[CL_type]  # Permet d'appeler un module avec une string
    classifierModule = getattr(classifierPackage, CL_type)
    classifierClass = getattr(classifierModule, CL_type)
    analysisModule = getattr(classifierPackage, "analyzeResults")

    logging.info("Start:\t Determine validation split for ratio " + str(learningRate))
    iValidationIndices = [DB.splitDataset(DATASET, learningRate, datasetLength) for iterIndex in range(statsIter)]
    iLearningIndices = [[index for index in range(datasetLength) if index not in validationIndices] for validationIndices in iValidationIndices]
    iClassificationSetLength = [len(learningIndices) for learningIndices in iLearningIndices]
    logging.info("Done:\t Determine validation split")

    logging.info("Start:\t Determine "+str(nbFolds)+" folds")
    if nbFolds != 1:
        iKFolds = [DB.getKFoldIndices(nbFolds, DATASET.get("Labels")[...], NB_CLASS, learningIndices) for learningIndices in iLearningIndices]
    else:
        iKFolds = [[[], range(classificationSetLength)] for classificationSetLength in iClassificationSetLength]

        # logging.info("Info:\t Length of Learning Sets: " + str(classificationSetLength - len(kFolds[0])))
        # logging.info("Info:\t Length of Testing Sets: " + str(len(kFolds[0])))
        # logging.info("Info:\t Length of Validation Set: " + str(len(validationIndices)))
        # logging.info("Done:\t Determine folds")


        # logging.info("Start:\t Learning with " + CL_type + " and " + str(len(kFolds)) + " folds")
        # logging.info("Start:\t Classification")
        # Begin Classification
    if hyperParamSearch != "None":
        classifier = searchBestSettings(DATASET, CL_type, metrics, iLearningIndices, iKFolds, viewsIndices=viewsIndices, searchingTool=hyperParamSearch, nIter=nIter, **classificationKWARGS)
    else:
        classifier = classifierClass(NB_CORES=nbCores, **classificationKWARGS)
        # classifier.setParams(classificationKWARGS)
    for _ in range(statsIter):
        classifier.fit_hdf5(DATASET, trainIndices=learningIndices, viewsIndices=viewsIndices)
        trainLabels = classifier.predict_hdf5(DATASET, usedIndices=learningIndices, viewsIndices=viewsIndices)
        testLabels = classifier.predict_hdf5(DATASET, usedIndices=validationIndices, viewsIndices=viewsIndices)
        fullLabels = classifier.predict_hdf5(DATASET, viewsIndices=viewsIndices)
        trainLabelsIterations.append(trainLabels)
        testLabelsIterations.append(testLabels)
        ivalidationIndices.append(validationIndices)
        classifiersIterations.append(classifier)
        logging.info("Done:\t Classification")

    classificationTime = time.time() - t_start

    logging.info("Info:\t Time for Classification: " + str(int(classificationTime)) + "[s]")
    logging.info("Start:\t Result Analysis for " + CL_type)

    times = (extractionTime, classificationTime)

    stringAnalysis, imagesAnalysis, metricsScores = analysisModule.execute(classifiersIterations, trainLabelsIterations,
                                                                           testLabelsIterations, DATASET,
                                                                           classificationKWARGS, learningRate,
                                                                           LABELS_DICTIONARY, views, nbCores, times,
                                                                           name, nbFolds, ivalidationIndices,
                                                                           hyperParamSearch, nIter, metrics, statsIter,
                                                                           viewsIndices)
    labelsSet = set(LABELS_DICTIONARY.values())
    logging.info(stringAnalysis)
    featureString = "-".join(views)
    labelsString = "-".join(labelsSet)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    CL_type_string = CL_type
    if CL_type=="Fusion":
        CL_type_string += "-"+classificationKWARGS["fusionType"]+"-"+classificationKWARGS["fusionMethod"]+"-"+"-".join(classificationKWARGS["classifiersNames"])
    elif CL_type=="Mumbo":
        CL_type_string += "-"+"-".join(classificationKWARGS["classifiersNames"])
    outputFileName = directory + timestr + "Results-" + CL_type_string + "-" + featureString + '-' + labelsString + \
                     '-learnRate' + str(learningRate) + '-' + name

    outputTextFile = open(outputFileName + '.txt', 'w')
    outputTextFile.write(stringAnalysis)
    outputTextFile.close()

    if imagesAnalysis is not None:
        for imageName in imagesAnalysis:
            if os.path.isfile(outputFileName + imageName + ".png"):
                for i in range(1,20):
                    testFileName = outputFileName + imageName + "-" + str(i) + ".png"
                    if os.path.isfile(testFileName )!=True:
                        imagesAnalysis[imageName].savefig(testFileName)
                        break

            imagesAnalysis[imageName].savefig(outputFileName + imageName + '.png')

    logging.info("Done:\t Result Analysis")
    return CL_type, classificationKWARGS, metricsScores, fullLabels