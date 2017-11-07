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
    feat = X.attrs["name"]
    CL_type = kwargs["CL_type"]
    X = getValue(X)
    learningRate = len(classificationIndices[0]) / (len(classificationIndices[0]) + len(classificationIndices[1]))
    labelsString = "-".join(labelsNames)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    CL_type_string = CL_type

    outputFileName = directory + "/" + CL_type_string + "/" + "/" + feat + "/" + timestr + "Results-" + CL_type_string + "-" + labelsString + \
                     '-learnRate' + str(learningRate) + '-' + name + "-" + feat + "-"
    if not os.path.exists(os.path.dirname(outputFileName)):
        try:
            os.makedirs(os.path.dirname(outputFileName))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return kwargs, t_start, feat, CL_type, X, learningRate, labelsString, timestr, outputFileName

def initTrainTest(X, Y, classificationIndices):
    trainIndices, testIndices = classificationIndices
    X_train = extractSubset(X, trainIndices)
    X_test = extractSubset(X, testIndices)
    y_train = Y[trainIndices]
    y_test = Y[testIndices]
    return X_train, y_train, X_test, y_test

def getKWARGS(classifierModule, hyperParamSearch, nIter, CL_type, X_train, y_train, randomState,
              outputFileName, KFolds, nbCores, metrics, kwargs):
    if hyperParamSearch != "None":
        classifierHPSearch = getattr(classifierModule, hyperParamSearch)
        logging.debug("Start:\t RandomSearch best settings with " + str(nIter) + " iterations for " + CL_type)
        cl_desc = classifierHPSearch(X_train, y_train, randomState, outputFileName, KFolds=KFolds, nbCores=nbCores,
                                     metric=metrics[0], nIter=nIter)
        clKWARGS = dict((str(index), desc) for index, desc in enumerate(cl_desc))
        logging.debug("Done:\t RandomSearch best settings")
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
    kwargs = args["args"]
    views = [DATASET.get("View" + str(viewIndex)).attrs["name"] for viewIndex in
             range(DATASET.get("Metadata").attrs["nbView"])]
    neededViewIndex = views.index(kwargs["feat"])
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
    timestr, \
    outputFileName = initConstants(args, X, classificationIndices, labelsNames, name, directory)
    logging.debug("Done:\t Loading data")

    logging.debug("Info:\t Classification - Database:" + str(name) + " Feature:" + str(feat) + " train ratio:"
                  + str(learningRate) + ", CrossValidation k-folds: " + str(KFolds.n_splits) + ", cores:"
                  + str(nbCores) + ", algorithm : " + CL_type)

    logging.debug("Start:\t Determine Train/Test split")
    X_train, y_train, X_test, y_test = initTrainTest(X, Y, classificationIndices)
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
    full_labels_pred = cl_res.predict(X)
    y_train_pred = cl_res.predict(X[classificationIndices[0]])
    y_test_pred = cl_res.predict(X[classificationIndices[1]])
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
    return viewIndex, [CL_type, cl_desc + [feat], metricsScores, full_labels_pred, clKWARGS]


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser(
    #     description='This methods permits to execute a multiclass classification with one single view. At this point the used classifier is a RandomForest. The GridSearch permits to vary the number of trees and CrossValidation with k-folds. The result will be a plot of the score per class and a CSV with the best classifier found by the GridSearch.',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    # groupStandard = parser.add_argument_group('Standard arguments')
    # groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
    # groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of Dataset', default=".hdf5")
    # groupStandard.add_argument('--name', metavar='STRING', action='store',
    #                            help='Name of Database (default: %(default)s)', default='DB')
    # groupStandard.add_argument('--feat', metavar='STRING', action='store',
    #                            help='Name of Feature for Classification (default: %(default)s)', default='RGB')
    # groupStandard.add_argument('--pathF', metavar='STRING', action='store',
    #                            help='Path to the views (default: %(default)s)', default='Results-FeatExtr/')
    # groupStandard.add_argument('--fileCL', metavar='STRING', action='store',
    #                            help='Name of classLabels CSV-file  (default: %(default)s)', default='classLabels.csv')
    # groupStandard.add_argument('--fileCLD', metavar='STRING', action='store',
    #                            help='Name of classLabels-Description CSV-file  (default: %(default)s)',
    #                            default='classLabels-Description.csv')
    # groupStandard.add_argument('--fileFeat', metavar='STRING', action='store',
    #                            help='Name of feature CSV-file  (default: %(default)s)', default='feature.csv')
    #
    # groupClass = parser.add_argument_group('Classification arguments')
    # groupClass.add_argument('--CL_type', metavar='STRING', action='store', help='Classifier to use',
    #                         default="RandomForest")
    # groupClass.add_argument('--CL_CV', metavar='INT', action='store', help='Number of k-folds for CV', type=int,
    #                         default=10)
    # groupClass.add_argument('--CL_Cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
    #                         default=1)
    # groupClass.add_argument('--CL_split', metavar='FLOAT', action='store', help='Split ratio for train and test',
    #                         type=float, default=0.9)
    # groupClass.add_argument('--CL_metrics', metavar='STRING', action='store',
    #                         help='Determine which metrics to use, separate with ":" if multiple, if empty, considering all',
    #                         default='')
    #
    # groupClassifier = parser.add_argument_group('Classifier Config')
    # groupClassifier.add_argument('--CL_config', metavar='STRING', nargs="+", action='store',
    #                              help='GridSearch: Determine the trees', default=['25:75:125:175'])
    #
    # args = parser.parse_args()
    #
    # classifierKWARGS = dict((key, value) for key, value in enumerate([arg.split(":") for arg in args.CL_config]))
    # ### Main Programm
    #
    #
    # # Configure Logger
    # directory = os.path.dirname(os.path.abspath(__file__)) + "/Results-ClassMonoView/"
    # logfilename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-LOG"
    # logfile = directory + logfilename
    # if os.path.isfile(logfile + ".log"):
    #     for i in range(1, 20):
    #         testFileName = logfilename + "-" + str(i) + ".log"
    #         if not os.path.isfile(directory + testFileName):
    #             logfile = directory + testFileName
    #             break
    # else:
    #     logfile += ".log"
    #
    # logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logfile, level=logging.DEBUG,
    #                     filemode='w')
    #
    # if args.log:
    #     logging.getLogger().addHandler(logging.StreamHandler())
    #
    # # Read the features
    # logging.debug("Start:\t Read " + args.type + " Files")
    #
    # if args.type == ".csv":
    #     X = np.genfromtxt(args.pathF + args.fileFeat, delimiter=';')
    #     Y = np.genfromtxt(args.pathF + args.fileCL, delimiter=';')
    # elif args.type == ".hdf5":
    #     dataset = h5py.File(args.pathF + args.name + ".hdf5", "r")
    #     viewsDict = dict((dataset.get("View" + str(viewIndex)).attrs["name"], viewIndex) for viewIndex in
    #                      range(dataset.get("Metadata").attrs["nbView"]))
    #     X = dataset["View" + str(viewsDict[args.feat])][...]
    #     Y = dataset["Labels"][...]
    #
    # logging.debug("Info:\t Shape of Feature:" + str(X.shape) + ", Length of classLabels vector:" + str(Y.shape))
    # logging.debug("Done:\t Read CSV Files")
    #
    # arguments = {args.CL_type + "KWARGS": classifierKWARGS, "feat": args.feat, "fileFeat": args.fileFeat,
    #              "fileCL": args.fileCL, "fileCLD": args.fileCLD, "CL_type": args.CL_type}
    # ExecMonoview(X, Y, args.name, args.CL_split, args.CL_CV, args.CL_Cores, args.type, args.pathF,
    #              metrics=args.CL_metrics, **arguments)
