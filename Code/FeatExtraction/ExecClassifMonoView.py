#!/usr/bin/env python

""" Script to perform a MonoView classification """

# Import built-in modules
import argparse                         # for acommand line arguments
import time                             # for time calculations
import datetime                         # for TimeStamp in CSVFile
import os                               # to geth path of the running script

# Import 3rd party modules
import numpy as np                      # for reading CSV-files and Series
import pandas as pd                     # for Series and DataFrames

# Import own modules
import ClassifMonoView	                # Functions for classification
import ExportResults                    # Functions to render results



# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-03-10

### Argument Parser
parser = argparse.ArgumentParser(
description='This methods permits to execute a multiclass classification with one single view. At this point the used classifier is a RandomForest. The GridSearch permits to vary the number of trees and CrossValidation with k-folds.', 
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('necessary arguments:')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)', default='Caltech')
groupStandard.add_argument('--feat', metavar='STRING', action='store', help='Name of Feature for Classification (default: %(default)s)', default='RGB')
groupStandard.add_argument('--pathF', metavar='STRING', action='store', help='Path to the features (default: %(default)s)', default='Results-FeatExtr\\')
groupStandard.add_argument('--fileCL', metavar='STRING', action='store', help='Name of classLabels CSV-file  (default: %(default)s)', default='D:\\classLabels.csv')
groupStandard.add_argument('--fileFeat', metavar='STRING', action='store', help='Name of feature CSV-file  (default: %(default)s)', default='D:\\feature.csv')

groupClass = parser.add_argument_group('Classification arguments:')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store', help='Determine the the train size', type=float, default=0.8)
groupClass.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='GridSearch: Determine the trees', default='50 100 150 200')
groupClass.add_argument('--CL_RF_CV', metavar='INT', action='store', help='Number of k-folds for CV', type=int, default=8)
groupClass.add_argument('--CL_RF_Cores', metavar='INT', action='store', help='Number of cores', type=int, default=1)

args = parser.parse_args()
num_estimators = map(int, args.CL_RF_trees.split())

### Main Programm

t_start = time.time()

# Determine the Database to extract features
print "### Main Programm for Classification MonoView"
print"### Info: Database:" + str(args.name) + " Feature:" + str(args.feat) + " train_size:" + str(args.CL_split) + ", GridSearch of Trees:" + args.CL_RF_trees + ", CrossValidation k-folds:" + str(args.CL_RF_CV) + ", cores:" + str(args.CL_RF_Cores)

# Einlesen von Features
print "### Start:\t Read CSV Files"

X = np.genfromtxt(args.pathF + args.fileFeat, delimiter=';')
Y = np.genfromtxt(args.pathF + args.fileCL, delimiter=';')

print "### Info:\t Shape of Feature:" + str(X.shape) + ", Length of classLabels vector:" + str(Y.shape)
print "### Done:\t Read CSV Files"

# Calculate Train/Test data
print "### Start:\t Determine Train/Test split"

X_train, X_test, y_train, y_test = ClassifMonoView.calcTrainTest(X, Y, args.CL_split)

print "### Info:\t Shape X_train:" + str(X_train.shape) + ", Length of y_train:" + str(len(y_train))
print "### Info:\t Shape X_test:" + str(X_test.shape) + ", Length of y_test:" + str(len(y_test))
print "### Done:\t Determine Train/Test split"

# Begin Classification RandomForest
print "### Start:\t Classification"

cl_desc, cl_res = ClassifMonoView.calcClassifRandomForestCV(X_train, y_train, num_estimators, args.CL_RF_CV, args.CL_RF_Cores)
t_end  = time.time() - t_start

# Add result to Results DF
df_class_res = pd.DataFrame()
df_class_res = df_class_res.append({'a_class_time':t_end, 'b_cl_desc': cl_desc, 'c_cl_res': cl_res, 
                                                'd_cl_score': cl_res.best_score_}, ignore_index=True)

print "### Info:\t Time for Classification: " + str(t_end) + "[s]"
print "### End:\t Classification"

# CSV Export
print "### Start:\t Exporting to CSV"
dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-ClassMonoView/"
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-CMV-" + args.name + "-" + args.feat
ExportResults.exportPandasToCSV(df_class_res, dir, filename)
print "### Done:\t Exporting to CSV"

# Plot Result
print "### Start:\t Plot Result"
np_score = ExportResults.calcScorePerClass(y_test, cl_res.predict(X_test).astype(int))
filename = dir + filename
ExportResults.showResults(filename, np_score)
print "### Done:\t Plot Result"



