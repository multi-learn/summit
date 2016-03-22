#!/usr/bin/env python

""" Script whichs helps to replot results from Feature Parameter Optimisation """

# Import built-in modules
import datetime                         # for TimeStamp in CSVFile
import argparse                         # for acommand line arguments
import os                               # to geth path of the running script

# Import 3rd party modules
import pandas as pd                     # for Series
import numpy as np                      # for DataFrames

# Import own modules
import ExportResults                    # Functions to render results

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25


parser = argparse.ArgumentParser(
description='This methods permits to execute a multiclass classification with one single view. At this point the used classifier is a RandomForest. The GridSearch permits to vary the number of trees and CrossValidation with k-folds.', 
formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()
args.valueEnd = 5
args.valueStart =75
args.nCalcs = 8
args.feature = "HOG"
args.param = "HOG_Cluster"
df_feat_res = pd.DataFrame.from_csv(path="D:\\BitBucket\\multiview-machine-learning-omis\\Code\\FeatExtraction\\Results-FeatParaOpt\\2016_03_19-FeatParaOpt-HOG.csv", sep=';')

# Get data from result to show results in plot
# Total time for feature extraction and classification
tot_time = df_feat_res.b_feat_extr_time.values + df_feat_res.e_cl_time.values
tot_time = np.asarray(tot_time)
# Time for feature extraction
feat_time = df_feat_res.b_feat_extr_time.values
feat_time = np.asarray(feat_time)
# Time for classification
cl_time = df_feat_res.e_cl_time.values
cl_time = np.asarray(cl_time)

# Mean Score of all classes
score = df_feat_res.f_cl_score.values
score = np.asarray(score)


# Range on X-Axis
if(args.nCalcs>1):
        step = float(args.valueEnd-args.valueStart)/float(args.nCalcs-1)
        rangeX = np.around(np.array(range(0,args.nCalcs))*step) + args.valueStart
else:
        rangeX = [args.valueStart]
rangeX = np.asarray(rangeX)

# Description of Classification
cl_desc = df_feat_res.c_cl_desc.values

# Description of Feature
feat_desc = df_feat_res.a_feat_desc.values

dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-FeatParaOpt/"

fileName = dir + datetime.datetime.now().strftime("%Y_%m_%d") + "-" + "Feature_" + args.feature + "-Parameter_" + args.param
store = False

# Show Results for Calculation
ExportResults.showScoreTime(fileName + "-TotalTime.png", store, score, tot_time, rangeX, args.param, feat_desc, cl_desc, 'Results for Parameter Optimisation', 'Precision', 'Total Time (Feature Extraction+Classification)\n [s]')
ExportResults.showScoreTime(fileName + "-FeatExtTime.png", store, score, feat_time, rangeX, args.param, feat_desc, cl_desc, 'Results for Parameter Optimisation', 'Precision', 'Feature Extraction Time\n [s]')
ExportResults.showScoreTime(fileName + "-ClassTime.png", store, score, cl_time, rangeX, args.param, feat_desc, cl_desc, 'Results for Parameter Optimisation', 'Precision', 'Classification Time\n [s]')