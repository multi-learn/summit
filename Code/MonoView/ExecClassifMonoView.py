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


# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25


### Argument Parser
parser = argparse.ArgumentParser(
description='This methods permits to execute a multiclass classification with one single view. At this point the used classifier is a RandomForest. The GridSearch permits to vary the number of trees and CrossValidation with k-folds. The result will be a plot of the score per class and a CSV with the best classifier found by the GridSearch.', 
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('Standard arguments')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
groupStandard.add_argument('--type', metavar='STRING', action='store', help='Type of Dataset', default="hdf5")
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


groupRF = parser.add_argument_group('Random Forest arguments')
groupRF.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='GridSearch: Determine the trees', default='25 75 125 175')

groupSVC = parser.add_argument_group('SVC arguments')
groupSVC.add_argument('--CL_SVC_kernel', metavar='STRING', action='store', help='GridSearch : Kernels used', default='linear')
groupSVC.add_argument('--CL_SVC_C', metavar='STRING', action='store', help='GridSearch : Penalty parameters used', default='1:10:100:1000')

groupRF = parser.add_argument_group('Decision Trees arguments')
groupRF.add_argument('--CL_DT_depth', metavar='STRING', action='store', help='GridSearch: Determine max depth for Decision Trees', default='1:3:5:7')

groupSGD = parser.add_argument_group('SGD arguments')
groupSGD.add_argument('--CL_SGD_alpha', metavar='STRING', action='store', help='GridSearch: Determine alpha for SGDClassifier', default='0.1:0.2:0.5:0.9')
groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store', help='GridSearch: Determine loss for SGDClassifier', default='log')
groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store', help='GridSearch: Determine penalty for SGDClassifier', default='l2')


args = parser.parse_args()
RandomForestKWARGS = {"classifier__n_estimators":map(int, args.CL_RF_trees.split())}
SVCKWARGS = {"classifier__kernel":args.CL_SVC_kernel.split(":"), "classifier__C":map(int,args.CL_SVC_C.split(":"))}
DecisionTreeKWARGS = {"classifier__max_depth":map(int,args.CL_DT_depth.split(":"))}
SGDKWARGS = {"classifier__alpha" : map(float,args.CL_SGD_alpha.split(":")), "classifier__loss":args.CL_SGD_loss.split(":"),
              "classifier__penalty":args.CL_SGD_penalty.split(":")}
### Main Programm
t_start = time.time()

# Configure Logger
dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-ClassMonoView/"
logfilename= datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-LOG"
logfile = dir + logfilename
if os.path.isfile(logfile + ".log"):
        for i in range(1,20):
                testFileName = logfilename  + "-" + str(i) + ".log"
                if os.path.isfile(dir + testFileName )!=True:
                        logfile = dir + testFileName
                        break
else:
        logfile = logfile + ".log"

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logfile, level=logging.DEBUG, filemode='w')

if(args.log):     
        logging.getLogger().addHandler(logging.StreamHandler())  

# Determine the Database to extract features
logging.debug("### Main Programm for Classification MonoView")
logging.debug("### Classification - Database:" + str(args.name) + " Feature:" + str(args.feat) + " train_size:" + str(args.CL_split) + ", GridSearch of Trees:" + args.CL_RF_trees + ", CrossValidation k-folds:" + str(args.CL_CV) + ", cores:" + str(args.CL_Cores))

# Read the features
logging.debug("Start:\t Read "+args.type+" Files")

if args.type == "csv":
    X = np.genfromtxt(args.pathF + args.fileFeat, delimiter=';')
    Y = np.genfromtxt(args.pathF + args.fileCL, delimiter=';')
elif args.type == "hdf5":
    dataset = h5py.File(args.pathF + args.name + ".hdf5", "r")
    viewsDict = dict((dataset.get("/View"+str(viewIndex)+"/name").value, viewIndex) for viewIndex in range(dataset.get("nbView").value))
    X = dataset["View"+str(viewsDict[args.feat])+"/matrix"][...]
    # X_ = dataset["View"+str(viewsDict[args.feat])+"/matrix"][...]
    # X = np.zeros((dataset.get("datasetLength/").value, dataset["View"+str(viewsDict[args.feat])+"/shape"][1]), dtype=int)
    # for exampleindice, exampleArray in enumerate(X_):
    #     dicti = dict((index, value) for index, value in enumerate(exampleArray))
    #     sorted_x = sorted(dicti.items(), key=operator.itemgetter(1))
    #     X[exampleindice] = np.array([index for (index, value) in sorted_x], dtype=int)

    Y = dataset["Labels/labelsArray"][...]

logging.debug("Info:\t Shape of Feature:" + str(X.shape) + ", Length of classLabels vector:" + str(Y.shape))
logging.debug("Done:\t Read CSV Files")

# Calculate Train/Test data
logging.debug("Start:\t Determine Train/Test split")

X_train, X_test, y_train, y_test = ClassifMonoView.calcTrainTest(X, Y, args.CL_split)

logging.debug("Info:\t Shape X_train:" + str(X_train.shape) + ", Length of y_train:" + str(len(y_train)))
logging.debug("Info:\t Shape X_test:" + str(X_test.shape) + ", Length of y_test:" + str(len(y_test)))
logging.debug("Done:\t Determine Train/Test split")

# Begin Classification RandomForest
logging.debug("Start:\t Classification")


classifierFunction = getattr(ClassifMonoView, "Classif"+args.CL_type)
classifierKWARGS = globals()[args.CL_type+"KWARGS"]

cl_desc, cl_res = classifierFunction(X_train, y_train, nbFolds=args.CL_CV, nbCores = args.CL_Cores,
                                                     **classifierKWARGS)
t_end  = time.time() - t_start

# Add result to Results DF
df_class_res = pd.DataFrame()
df_class_res = df_class_res.append({'a_class_time':t_end, 'b_cl_desc': cl_desc, 'c_cl_res': cl_res, 
                                                'd_cl_score': cl_res.best_score_}, ignore_index=True)

logging.debug("Info:\t Time for Classification: " + str(t_end) + "[s]")
logging.debug("Done:\t Classification")

# CSV Export
logging.debug("Start:\t Exporting to CSV")
dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-ClassMonoView/"
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat
ExportResults.exportPandasToCSV(df_class_res, dir, filename)
logging.debug("Done:\t Exporting to CSV")

# Stats Result
y_test_pred = cl_res.predict(X_test)
classLabelsDesc = pd.read_csv(args.pathF + args.fileCLD, sep=";", names=['label', 'name'])
classLabelsNames = classLabelsDesc.name
#logging.debug("" + str(classLabelsNames))
classLabelsNamesList = classLabelsNames.values.tolist()
#logging.debug(""+ str(classLabelsNamesList))

logging.debug("Start:\t Statistic Results")

#Accuracy classification score
accuracy_score = ExportResults.accuracy_score(y_test, y_test_pred)

# Classification Report with Precision, Recall, F1 , Support
logging.debug("Info:\t Classification report:")
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Report"
logging.debug("\n" + str(metrics.classification_report(y_test, y_test_pred, labels = range(0,len(classLabelsDesc.name)), target_names=classLabelsNamesList)))
scores_df = ExportResults.classification_report_df(dir, filename, y_test, y_test_pred, range(0, len(classLabelsDesc.name)), classLabelsNamesList)

# Create some useful statistcs
logging.debug("Info:\t Statistics:")
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Stats"
stats_df = ExportResults.classification_stats(dir, filename, scores_df, accuracy_score)
logging.debug("\n" + stats_df.to_string())

# Confusion Matrix
logging.debug("Info:\t Calculate Confusionmatrix")
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-ConfMatrix"
df_conf_norm = ExportResults.confusion_matrix_df(dir, filename, y_test, y_test_pred, classLabelsNamesList)
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-ConfMatrixImg"
ExportResults.plot_confusion_matrix(dir, filename, df_conf_norm)

logging.debug("Done:\t Statistic Results")


# Plot Result
logging.debug("Start:\t Plot Result")
np_score = ExportResults.calcScorePerClass(y_test, cl_res.predict(X_test).astype(int))
### dir and filename the same as CSV Export
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat + "-Score"
ExportResults.showResults(dir, filename, args.name, args.feat, np_score)
logging.debug("Done:\t Plot Result")


