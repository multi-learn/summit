#!/usr/bin/env python

""" Script whichs helps to replot results from Feature Parameter Optimisation """

# Import built-in modules
import argparse  # for acommand line arguments
import datetime  # for TimeStamp in CSVFile
import os  # to geth path of the running script
import matplotlib

# matplotlib.use('Agg')
# Import 3rd party modules
import pandas as pd  # for Series
import numpy as np  # for DataFrames

# Import own modules
import ExportResults  # Functions to render results

# Author-Info
__author__ = "Nikolas Huelsmann"
__status__ = "Prototype"  # Production, Development, Prototype
__date__ = 2016 - 03 - 25

parser = argparse.ArgumentParser(
    description='This method can be used to replot results from Feature Parameter Optimisation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()
args.name = "Caltech"
args.valueStart = 2
args.valueEnd = 25
args.nCalcs = 5
args.feature = "HSV"
args.param = "HSV_V_Bins"
args.show = False
df_feat_res = pd.DataFrame.from_csv(
    path="D:\\BitBucket\\multiview-machine-learning-omis\\Results\\Hydra\\2016_03_23-FPO-Caltech-HSV-HSV_V_Bins.csv",
    sep=';')

# Get data from result to show results in plot
# logging.debug("Start:\t Plot Result")
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
if args.nCalcs > 1:
    step = float(args.valueEnd - args.valueStart) / float(args.nCalcs - 1)
    rangeX = np.around(np.array(range(0, args.nCalcs)) * step) + args.valueStart
else:
    rangeX = [args.valueStart]
rangeX = np.asarray(rangeX)

# Description of Classification
cl_desc = df_feat_res.c_cl_desc.values

# Description of Feature
feat_desc = df_feat_res.a_feat_desc.values

dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-FeatParaOpt/"
# filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-FPO-" + args.name + "-" + args.feature + "-" + args.param
# ExportResults.exportPandasToCSV(df_feat_res, directory, filename)

# Store or Show plot
if args.show:
    store = False
else:
    store = True

fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-FPO-" + args.name + "-" + args.feature + "-" + args.param
# Show Results for Calculation
ExportResults.showScoreTime(dir, fileName + "-TotalTime", store, score, tot_time, rangeX, args.param, feat_desc,
                            cl_desc, 'Results for Parameter Optimisation - DB:' + args.name + ' Feat:' + args.feature,
                            'Precision', 'Total Time (Feature Extraction+Classification)\n [s]')
ExportResults.showScoreTime(dir, fileName + "-FeatExtTime", store, score, feat_time, rangeX, args.param, feat_desc,
                            cl_desc, 'Results for Parameter Optimisation - DB:' + args.name + ' Feat:' + args.feature,
                            'Precision', 'Feature Extraction Time\n [s]')
ExportResults.showScoreTime(dir, fileName + "-ClassTime", store, score, cl_time, rangeX, args.param, feat_desc, cl_desc,
                            'Results for Parameter Optimisation - DB:' + args.name + ' Feat:' + args.feature,
                            'Precision', 'Classification Time\n [s]')
