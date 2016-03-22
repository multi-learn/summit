#!/usr/bin/env python

""" Execution: Script to perform feature parameter optimisation """

# Import built-in modules
import datetime                         # for TimeStamp in CSVFile
import argparse                         # for acommand line arguments
import os                               # to geth path of the running script

# Import 3rd party modules
import numpy as np                      # for arrays
import logging                          # To create Log-Files  

# Import own modules
import DBCrawl			        # Functions to read Images from Database
import FeatParaOpt		        # Functions to perform parameter optimisation
import ExportResults                    # Functions to render results

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25

### Argument Parser

parser = argparse.ArgumentParser(
description='This methods permits to perform an optimisation of the parameter of one feature. Therefore you have so specify which feature to use (e.g. --feature RGB) and which of his parameters (the parameters depend on the feature chosen, e.g. for RGB: --parameter Bins). The method will calculate the results in your given range and export the results to a CSV-File.',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

groupStandard = parser.add_argument_group('necessary arguments:')
groupStandard.add_argument('--name', metavar='STRING', action='store', help='Select a name of DB, e.g. Caltech (default: %(default)s)', default='DB')
groupStandard.add_argument('--path', metavar='STRING', action='store', help='Path to the database (default: %(default)s)', default='D:\\CaltechMini')
groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')

groupOpt = parser.add_argument_group('Optimisation arguments:')
groupOpt.add_argument('--feature', choices=['RGB', 'HSV', 'SURF', 'SIFT', 'HOG'], help='Set feature from list (RGB, HSV, ..)', default='RGB')
groupOpt.add_argument('--param', choices=['RGB_Bins', 'RGB_MaxCI', 'HSV_H_Bins', 'HSV_S_Bins', 'HSV_V_Bins', 'SIFT_Cluster', 'SURF_Cluster', 'HOG_Cluster'], help='Parameter to optimise (remember depends on feature)', default='RGB_Bins')
groupOpt.add_argument('--valueStart', metavar='INT', action='store', help='Start-Value for optimisation range', type=int)
groupOpt.add_argument('--valueEnd', metavar='INT', action='store', help='End-Value for optimisation range', type=int)
groupOpt.add_argument('--nCalcs', metavar='INT', action='store', help='Number of calculations between Start and End-Value', type=int)

groupRGB = parser.add_argument_group('RGB arguments:')
groupRGB.add_argument('--RGB_Bins', metavar='INT', action='store', help='Number of bins for histogram', type=int, default=16)
groupRGB.add_argument('--RGB_CI', metavar='INT', action='store', help='Max Color Intensity [0 to VALUE]', type=int, default=256)
groupRGB.add_argument('-RGB_NMinMax', action='store_true', help='Use option to actvate MinMax Norm instead of Distribution')

groupHSV = parser.add_argument_group('HSV arguments:')
groupHSV.add_argument('--HSV_H_Bins', metavar='INT', action='store', help='Number of bins for Hue', type=int, default=16)
groupHSV.add_argument('--HSV_S_Bins', metavar='INT', action='store', help='Number of bins for Saturation', type=int, default=4)
groupHSV.add_argument('--HSV_V_Bins', metavar='INT', action='store', help='Number of bins for Value', type=int, default=4)
groupHSV.add_argument('-HSV_NMinMax', action='store_true', help='Use option to actvate MinMax Norm instead of Distribution')

groupSIFT = parser.add_argument_group('SIFT arguments:')
groupSIFT.add_argument('--SIFT_Cluster', metavar='INT', action='store', help='Number of k-means cluster', type=int, default=50)
groupSIFT.add_argument('-SIFT_NMinMax', action='store_true', help='Use option to actvate MinMax Norm instead of Distribution')
        
groupSURF = parser.add_argument_group('SURF arguments:')
groupSURF.add_argument('--SURF_Cluster', metavar='INT', action='store', help='Number of k-means cluster', type=int, default=50)
groupSURF.add_argument('-SURF_NMinMax', action='store_true', help='Use option to actvate MinMax Norm instead of Distribution')

groupHOG = parser.add_argument_group('HOG arguments:')
groupHOG.add_argument('--HOG_CellD', metavar='INT', action='store', help='CellDimension for local histograms', type=int, default=5)
groupHOG.add_argument('--HOG_Orient', metavar='INT', action='store', help='Number of bins of local histograms', type=int, default=8)
groupHOG.add_argument('--HOG_Cluster', metavar='INT', action='store', help='Number of k-means cluster', type=int, default=12)
groupHOG.add_argument('--HOG_Iter', metavar='INT', action='store', help='Max. number of iterations for clustering', type=int, default=100)
groupHOG.add_argument('--HOG_cores', metavar='INT', action='store', help='Number of cores for HOG', type=int, default=1)

groupClass = parser.add_argument_group('Classification arguments:')
groupClass.add_argument('--CL_split', metavar='FLOAT', action='store', help='Determine the train size', type=float, default=0.8)
groupClass.add_argument('--CL_RF_trees', metavar='STRING', action='store', help='GridSearch: Determine the trees', default='50 100 150 200')
groupClass.add_argument('--CL_RF_CV', metavar='INT', action='store', help='Number of k-folds for CV', type=int, default=8)
groupClass.add_argument('--CL_RF_Cores', metavar='INT', action='store', help='Number of cores', type=int, default=1)



### Read args - transform in Arrays for function calls
args = parser.parse_args()
path = args.path
nameDB = args.name

para_opt = [args.feature, args.param, args.valueStart, args.valueEnd, args.nCalcs]
para_RGB = [args.RGB_Bins, args.RGB_CI, args.RGB_NMinMax]
para_HSV = [args.HSV_H_Bins, args.HSV_S_Bins, args.HSV_V_Bins, args.HSV_NMinMax]
para_SIFT = [args.SIFT_Cluster, args.SIFT_NMinMax]
para_SURF = [args.SURF_Cluster, args.SURF_NMinMax]
para_HOG = [args.HOG_CellD, args.HOG_Orient, args.HOG_Cluster, args.HOG_Iter, args.HOG_cores]

para_Cl = [args.CL_split, map(int, args.CL_RF_trees.split()), args.CL_RF_CV, args.CL_RF_Cores]

       
### Main Programm

# Configure Logger
dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-FeatParaOpt/"
logfilename= datetime.datetime.now().strftime("%Y_%m_%d") + "-LOG-FPO-" + args.name + "-" + args.feat + "-" + args.param
logfile = dir + logfilename
if os.path.isfile(logfile + ".log"):
        for i in range(1,20):
                testFileName = logfilename  + "-" + str(i) + ".log"
                if os.path.isfile(dir + testFileName )!=True:
                        logfile = dir + testFileName
                        break
else:
        logfile = logfile + ".log"

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', filename=logfile, level=logging.DEBUG, filemode='w')

if(args.log):     
        logging.getLogger().addHandler(logging.StreamHandler())  

################################ Read Images from Database
# Determine the Database to extract features

logging.debug("### Main Programm for Feature Parameter Optimisation ")
logging.debug('### Optimisation - Feature:' + str(args.feature) + " Parameter:" + str(args.param) + " from:" + str(args.valueStart) + " to:" + str(args.valueEnd) + " in #calc:" + str(args.nCalcs))

logging.debug("### Start:\t Exportation of images from DB")

# get dictionary to link classLabels Text to Integers
sClassLabels = DBCrawl.getClassLabels(path)

# Get all path from all images inclusive classLabel as Integer
dfImages,nameDB = DBCrawl.imgCrawl(path, sClassLabels, nameDB)

logging.debug("### Done:\t Exportation of Images from DB ")


################################ Parameter Optimisation
logging.debug("### Start:\t Feautre Optimisation")
df_feat_res = FeatParaOpt.perfFeatMonoV(nameDB, dfImages, para_opt, para_RGB, para_HSV, para_SIFT, para_SURF, para_HOG, para_Cl)
logging.debug("### Done:\t Feautre Optimisation ")


################################ Render results
logging.debug("### Start:\t Exporting to CSV ")
dir = os.path.dirname(os.path.abspath(__file__)) + "/Results-FeatParaOpt/"
filename = datetime.datetime.now().strftime("%Y_%m_%d") + "-FeatParaOpt-" + args.feature
ExportResults.exportPandasToCSV(df_feat_res, dir, filename)
logging.debug("### Done:\t Exporting to CSV ")

# Get data from result to show results in plot
logging.debug("### Start:\t Plot Result")
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
store = True

fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + "-FPO-" + args.name + "-" + args.feature + "-" + args.param
# Show Results for Calculation
ExportResults.showScoreTime(dir, fileName + "-TotalTime.png", store, score, tot_time, rangeX, args.param, feat_desc, cl_desc, 'Results for Parameter Optimisation \n DB: ' + args.name + 'Feat: ' + args.feature, 'Precision', 'Total Time (Feature Extraction+Classification)\n [s]')
ExportResults.showScoreTime(dir, fileName + "-FeatExtTime.png", store, score, feat_time, rangeX, args.param, feat_desc, cl_desc, 'Results for Parameter Optimisation \n DB: ' + args.name + 'Feat: ' + args.feature, 'Precision', 'Feature Extraction Time\n [s]')
ExportResults.showScoreTime(dir, fileName + "-ClassTime.png", store, score, cl_time, rangeX, args.param, feat_desc, cl_desc, 'Results for Parameter Optimisation \n DB: ' + args.name + 'Feat: ' + args.feature, 'Precision', 'Classification Time\n [s]')

logging.debug("### Done:\t Plot Result")


#print 'Les meilleurs parametres sont: ' + str(rf_detector.best_params_)

#print '\nLe meilleure score avec ces parametres est: ' + str(rf_detector.best_score_)

#print '\n Les resultas pour tous les parametres avec GridSearch: \n'
#print lr_detector.grid_scores_	

#get_ipython().magic(u'time forest = forest.fit(data_train, label_train)')
#print 'RandomForest with ' + str(num_estimators) + " Trees: " + str(forest.score(data_test, label_test))


