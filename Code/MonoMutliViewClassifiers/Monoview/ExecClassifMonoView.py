#!/usr/bin/env python

""" Execution: Script to perform a MonoView classification """

# Import built-in modules
import argparse                         # for command line arguments
import datetime                         # for TimeStamp in CSVFile
import os                               # to geth path of the running script
import time                             # for time calculations
import operator

# Import 3rd party modules
import numpy as np                      # for reading CSV-files and Series
import pandas as pd                     # for Series and DataFrames
import logging                          # To create Log-Files
from sklearn import metrics		# For stastics on classification
import h5py

# Import own modules
import ClassifMonoView	                # Functions for classification
import ExportResults                    # Functions to render results
import MonoviewClassifiers
import Metrics
from analyzeResult import execute

# Author-Info
__author__ 	= "Nikolas Huelsmann, Baptiste BAUVIN"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25


def ExecMonoview_multicore(name, learningRate, nbFolds, datasetFileIndex, databaseType, path, gridSearch=True,
                           metrics=[["accuracy_score", None]], nIter=30, **args):
    DATASET = h5py.File(path+name+str(datasetFileIndex)+".hdf5", "r")
    kwargs = args["args"]

    views = [DATASET.get("View"+str(viewIndex)).attrs["name"] for viewIndex in range(DATASET.get("Metadata").attrs["nbView"])]
    neededViewIndex = views.index(kwargs["feat"])
    X = DATASET.get("View"+str(neededViewIndex))
    Y = DATASET.get("labels").value
    returnedViewIndex = args["viewIndex"]
    return ExecMonoview(X, Y, name, learningRate, nbFolds, 1, databaseType, path, gridSearch=gridSearch,
                        metrics=metrics, nIter=nIter, **args)


def ExecMonoview(X, Y, name, learningRate, nbFolds, nbCores, databaseType, path, gridSearch=True,
                metrics=[["accuracy_score", None]], nIter=30, **args):

    try:
        kwargs = args["args"]
    except:
        kwargs = args
    t_start = time.time()
    directory = os.path.dirname(os.path.abspath(__file__)) + "/Results-ClassMonoView/"
    feat = X.attrs["name"]
    fileFeat = kwargs["fileFeat"]
    fileCL = kwargs["fileCL"]
    fileCLD = kwargs["fileCLD"]
    CL_type = kwargs["CL_type"]
    X = X.value
    clKWARGS = kwargs[kwargs["CL_type"]+"KWARGS"]

    # Determine the Database to extract features
    logging.debug("### Main Programm for Classification MonoView")
    logging.debug("### Classification - Database:" + str(name) + " Feature:" + str(feat) + " train_size:" + str(learningRate) + ", CrossValidation k-folds:" + str(nbFolds) + ", cores:" + str(nbCores)+", algorithm : "+CL_type)


    # Calculate Train/Test data
    logging.debug("Start:\t Determine Train/Test split")

    X_train, X_test, y_train, y_test = ClassifMonoView.calcTrainTest(X, Y, learningRate)

    logging.debug("Info:\t Shape X_train:" + str(X_train.shape) + ", Length of y_train:" + str(len(y_train)))
    logging.debug("Info:\t Shape X_test:" + str(X_test.shape) + ", Length of y_test:" + str(len(y_test)))
    logging.debug("Done:\t Determine Train/Test split")

    # Begin Classification RandomForest

    classifierModule = getattr(MonoviewClassifiers, CL_type)
    classifierGridSearch = getattr(classifierModule, "gridSearch")

    if gridSearch:
        logging.debug("Start:\t RandomSearch best settings with "+str(nIter)+" iterations")
        cl_desc = classifierGridSearch(X_train, y_train, nbFolds=nbFolds, nbCores=nbCores, metric=metrics[0], nIter=nIter)
        clKWARGS = dict((str(index), desc) for index, desc in enumerate(cl_desc))
        logging.debug("Done:\t RandomSearch best settings")
    logging.debug("Start:\t Training")
    cl_res = classifierModule.fit(X_train, y_train, NB_CORES=nbCores, **clKWARGS)
    t_end  = time.time() - t_start

    logging.debug("Info:\t Time for Training: " + str(t_end) + "[s]")
    logging.debug("Done:\t Training")

    logging.debug("Start:\t Predicting")
    # Stats Result
    y_train_pred = cl_res.predict(X_train)
    y_test_pred = cl_res.predict(X_test)
    classLabelsDesc = pd.read_csv(path + fileCLD, sep=";", names=['label', 'name'])
    classLabelsNames = classLabelsDesc.name
    logging.debug("Done:\t Predicting")
    #logging.debug("" + str(classLabelsNames))
    classLabelsNamesList = classLabelsNames.values.tolist()
    #logging.debug(""+ str(classLabelsNamesList))

    logging.debug("Start:\t Getting Results")

    #Accuracy classification score
    stringAnalysis, imagesAnalysis, train, ham, test = execute(name, learningRate, nbFolds, nbCores, gridSearch, metrics, nIter, feat, CL_type,
                                         clKWARGS, classLabelsNames, X.shape,
                                         y_train, y_train_pred, y_test, y_test_pred, t_end)
    cl_desc = [value for key, value in sorted(clKWARGS.iteritems())]
    logging.debug("Done:\t Getting Results")
    logging.info(stringAnalysis)
    labelsString = "-".join(classLabelsNames)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    CL_type_string = CL_type
    outputFileName = "Results/" + timestr + "Results-" + CL_type_string + "-" + labelsString + \
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
    viewIndex = args["viewIndex"]
    return viewIndex, [CL_type, test, cl_desc, feat]
    # # Classification Report with Precision, Recall, F1 , Support
    # logging.debug("Info:\t Classification report:")
    # filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + name + "-" + feat + "-Report"
    # logging.debug("\n" + str(metrics.classification_report(y_test, y_test_pred, labels = range(0,len(classLabelsDesc.name)), target_names=classLabelsNamesList)))
    # scores_df = ExportResults.classification_report_df(directory, filename, y_test, y_test_pred, range(0, len(classLabelsDesc.name)), classLabelsNamesList)
    #
    # # Create some useful statistcs
    # logging.debug("Info:\t Statistics:")
    # filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + name + "-" + feat + "-Stats"
    # stats_df = ExportResults.classification_stats(directory, filename, scores_df, accuracy_score)
    # logging.debug("\n" + stats_df.to_string())
    #
    # # Confusion Matrix
    # logging.debug("Info:\t Calculate Confusionmatrix")
    # filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + name + "-" + feat + "-ConfMatrix"
    # df_conf_norm = ExportResults.confusion_matrix_df(directory, filename, y_test, y_test_pred, classLabelsNamesList)
    # filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + name + "-" + feat + "-ConfMatrixImg"
    # ExportResults.plot_confusion_matrix(directory, filename, df_conf_norm)
    #
    # logging.debug("Done:\t Statistic Results")
    #
    #
    # # Plot Result
    # logging.debug("Start:\t Plot Result")
    # np_score = ExportResults.calcScorePerClass(y_test, cl_res.predict(X_test).astype(int))
    # ### directory and filename the same as CSV Export
    # filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + name + "-" + feat + "-Score"
    # ExportResults.showResults(directory, filename, name, feat, np_score)
    # logging.debug("Done:\t Plot Result")
    # return [CL_type, accuracy_score, cl_desc]


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='This methods permits to execute a multiclass classification with one single view. At this point the used classifier is a RandomForest. The GridSearch permits to vary the number of trees and CrossValidation with k-folds. The result will be a plot of the score per class and a CSV with the best classifier found by the GridSearch.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
    groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of Dataset', default=".hdf5")
    groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)', default='DB')
    groupStandard.add_argument('--feat', metavar='STRING', action='store', help='Name of Feature for Classification (default: %(default)s)', default='RGB')
    groupStandard.add_argument('--pathF', metavar='STRING', action='store', help='Path to the views (default: %(default)s)', default='Results-FeatExtr/')
    groupStandard.add_argument('--fileCL', metavar='STRING', action='store', help='Name of classLabels CSV-file  (default: %(default)s)', default='classLabels.csv')
    groupStandard.add_argument('--fileCLD', metavar='STRING', action='store', help='Name of classLabels-Description CSV-file  (default: %(default)s)', default='classLabels-Description.csv')
    groupStandard.add_argument('--fileFeat', metavar='STRING', action='store', help='Name of feature CSV-file  (default: %(default)s)', default='feature.csv')


    groupClass = parser.add_argument_group('Classification arguments')
    groupClass.add_argument('--CL_type', metavar='STRING', action='store', help='Classifier to use', default="RandomForest")
    groupClass.add_argument('--CL_CV', metavar='INT', action='store', help='Number of k-folds for CV', type=int, default=10)
    groupClass.add_argument('--CL_Cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int, default=1)
    groupClass.add_argument('--CL_split', metavar='FLOAT', action='store', help='Split ratio for train and test', type=float, default=0.9)
    groupClass.add_argument('--CL_metrics', metavar='STRING', action='store',
                        help='Determine which metric to use, separate with ":" if multiple, if empty, considering all', default='')


    groupClassifier = parser.add_argument_group('Classifier Config')
    groupClassifier.add_argument('--CL_config', metavar='STRING', nargs="+", action='store', help='GridSearch: Determine the trees', default=['25:75:125:175'])

    # groupSVMLinear = parser.add_argument_group('SVC arguments')
    # groupSVMLinear.add_argument('--CL_SVML_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used', default='1:10:100:1000')
    #
    # groupSVMRBF = parser.add_argument_group('SVC arguments')
    # groupSVMRBF.add_argument('--CL_SVMR_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used', default='1:10:100:1000')
    #
    # groupRF = parser.add_argument_group('Decision Trees arguments')
    # groupRF.add_argument('--CL_DT_depth', metavar='STRING', action='store', help='GridSearch: Determine max depth for Decision Trees', default='1:3:5:7')
    #
    # groupSGD = parser.add_argument_group('SGD')
    # groupSGD.add_argument('--CL_SGD_alpha', metavar='STRING', action='store', help='GridSearch: Determine alpha for SGDClassifier', default='0.1:0.2:0.5:0.9')
    # groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store', help='GridSearch: Determine loss for SGDClassifier', default='log')
    # groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store', help='GridSearch: Determine penalty for SGDClassifier', default='l2')


    args = parser.parse_args()

    # RandomForestKWARGS = {"classifier__n_estimators":map(int, args.CL_RF_trees.split())}
    # SVMLinearKWARGS = {"classifier__C":map(int,args.CL_SVML_C.split(":"))}
    # SVMRBFKWARGS = {"classifier__C":map(int,args.CL_SVMR_C.split(":"))}
    # DecisionTreeKWARGS = {"classifier__max_depth":map(int,args.CL_DT_depth.split(":"))}
    # SGDKWARGS = {"classifier__alpha" : map(float,args.CL_SGD_alpha.split(":")), "classifier__loss":args.CL_SGD_loss.split(":"),
    #              "classifier__penalty":args.CL_SGD_penalty.split(":")}
    classifierKWARGS = dict((key, value) for key, value in enumerate([arg.split(":") for arg in args.CL_config]))
    ### Main Programm


    # Configure Logger
    directory = os.path.dirname(os.path.abspath(__file__)) + "/Results-ClassMonoView/"
    logfilename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-LOG"
    logfile = directory + logfilename
    if os.path.isfile(logfile + ".log"):
        for i in range(1,20):
            testFileName = logfilename  + "-" + str(i) + ".log"
            if os.path.isfile(directory + testFileName)!=True:
                logfile = directory + testFileName
                break
    else:
        logfile = logfile + ".log"

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logfile, level=logging.DEBUG, filemode='w')

    if(args.log):
        logging.getLogger().addHandler(logging.StreamHandler())


    # Read the features
    logging.debug("Start:\t Read " + args.type + " Files")

    if args.type == ".csv":
        X = np.genfromtxt(args.pathF + args.fileFeat, delimiter=';')
        Y = np.genfromtxt(args.pathF + args.fileCL, delimiter=';')
    elif args.type == ".hdf5":
        dataset = h5py.File(args.pathF + args.name + ".hdf5", "r")
        viewsDict = dict((dataset.get("View"+str(viewIndex)).attrs["name"], viewIndex) for viewIndex in range(dataset.get("Metadata").attrs["nbView"]))
        X = dataset["View"+str(viewsDict[args.feat])][...]
        Y = dataset["labels"][...]

    logging.debug("Info:\t Shape of Feature:" + str(X.shape) + ", Length of classLabels vector:" + str(Y.shape))
    logging.debug("Done:\t Read CSV Files")

    arguments = {args.CL_type+"KWARGS": classifierKWARGS, "feat":args.feat,"fileFeat": args.fileFeat,
                 "fileCL": args.fileCL, "fileCLD": args.fileCLD, "CL_type": args.CL_type}
    ExecMonoview(X, Y, args.name, args.CL_split, args.CL_CV, args.CL_Cores, args.type, args.pathF,
                 metrics=args.CL_metrics, **arguments)
