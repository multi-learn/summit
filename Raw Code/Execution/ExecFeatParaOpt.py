#!/usr/bin/env python

""" Script to perform feature parameter optimisation """

# Import built-in modules
import cv2                      # for OpenCV 
import cv                       # for OpenCV
import datetime                 # for TimeStamp in CSVFile
from scipy.cluster.vq import *  # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np              # for arrays

# Import sci-kit learn
from sklearn.ensemble import RandomForestClassifier

# Import own modules
import DBCrawl			# Functions to read Images from Database
import FeatParaOpt		# Functions to perform parameter optimisation
import ExportResults            # Functions to render results

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-01-23

### Main Programm

################################ Read Images from Database
# Determine the Database to extract features

print "### Start of Main Programm for Feature Parameter Optimisation ###"
path ="D:\\Caltech"
nameDB = "CT"

print "### Start:\t Exportation of images from DB ###"

# get dictionary to link classLabels Text to Integers
sClassLabels = DBCrawl.getClassLabels(path)

# Get all path from all images inclusive classLabel as Integer
dfImages,nameDB = DBCrawl.imgCrawl(path, sClassLabels, nameDB)

print "### Done:\t Exportation of Images from DB ###"


################################ Parameter Optimisation
# Setup
#feature = "RGB"
#parameter = "Bins"
#valueStart = int(8)
#valueEnd = int(64)
#nCalculations = int(8)
#boolCV = True

#print '### Optimisation - Feature:' + str(feature) + " Parameter:" + str(parameter) + " from:" + str(valueStart) + " to:" + str(valueEnd) + " in #calc:" + str(nCalculations) + " withCV:" + str(boolCV) + " ###"

#print "### Start: Feautre Optimisation ###"
#df_feat_res = FeatParaOpt.perfFeatMonoV(nameDB, dfImages,feature, parameter, valueStart, valueEnd, nCalculations, boolCV)
#print "### Done: Feautre Optimisation ###"

# Setup SURF
feature = "SURF"
parameter = "Cluster"
valueStart = 50
valueEnd  = 200
nCalculations = 4
boolCV = True

print '### Optimisation - Feature:' + str(feature) + " Parameter:" + str(parameter) + " from:" + str(valueStart) + " to:" + str(valueEnd) + " in #calc:" + str(nCalculations) + " withCV:" + str(boolCV) + " ###"                 

print "### Start:\t Feautre Optimisation ###"
df_feat_res = FeatParaOpt.perfFeatMonoV(nameDB, dfImages,feature, parameter, valueStart, valueEnd, nCalculations, boolCV)
print "### Done:\t Feautre Optimisation ###"

################################ Render results
print "### Start:\t Exporting to CSV ###"
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-Results-" + feature
ExportResults.exportPandasToCSV(df_feat_res, filename)
print "### Done:\t Exporting to CSV ###"

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
if(nCalculations>1):
        step = float(valueEnd-valueStart)/float(nCalculations-1)
        rangeX = np.around(np.array(range(0,nCalculations))*step) + valueStart
else:
        rangeX = [valueStart]
rangeX = np.asarray(rangeX)

# Description of Classification
cl_desc = df_feat_res.c_cl_desc.values

# Description of Feature
feat_desc = df_feat_res.a_feat_desc.values

# Show Results for Calculation
ExportResults.showScoreTime(score, tot_time, rangeX, parameter, feat_desc, cl_desc, 'Results for Parameter Optimisation', 'Precision', 'Total Time (Feature Extraction+Classification)\n [s]')
ExportResults.showScoreTime(score, feat_time, rangeX, parameter, feat_desc, cl_desc, 'Results for Parameter Optimisation', 'Precision', 'Feature Extraction Time\n [s]')
ExportResults.showScoreTime(score, cl_time, rangeX, parameter, feat_desc, cl_desc, 'Results for Parameter Optimisation', 'Precision', 'Classification Time\n [s]')



#print 'Les meilleurs parametres sont: ' + str(rf_detector.best_params_)

#print '\nLe meilleure score avec ces parametres est: ' + str(rf_detector.best_score_)

#print '\n Les resultas pour tous les parametres avec GridSearch: \n'
#print lr_detector.grid_scores_	

#get_ipython().magic(u'time forest = forest.fit(data_train, label_train)')
#print 'RandomForest with ' + str(num_estimators) + " Trees: " + str(forest.score(data_test, label_test))


