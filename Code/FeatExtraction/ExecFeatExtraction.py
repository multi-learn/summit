#!/usr/bin/env python

""" Script to perform feature parameter optimisation """

# Import built-in modules
import cv2                      # for OpenCV 
import cv                       # for OpenCV
import datetime                 # for TimeStamp in CSVFile
from scipy.cluster.vq import *  # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np              # for arrays
import time                     # for time calculations
from argparse import ArgumentParser # for acommand line arguments

# Import 3rd party modules

# Import own modules
import DBCrawl			# Functions to read Images from Database
import ExportResults            # Functions to render results
import FeatExtraction           # Functions to extract the features from Database   

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-03-10

### Argument Parser

parser = ArgumentParser(description='Export Features')

parser.add_argument('--name', action='store', help='Name of DB, default DB', default='DB')
parser.add_argument('--path', action='store', help='Path to the database e.g. D:\\Caltech', default='D:\\CaltechMini')
parser.add_argument('--cores', action='store', help='Number of cores used for parallelization of HOG, default 1', type=int, default=1)

parser.add_argument('--RGB', action='store_true', help='Use option to activate RGB')
parser.add_argument('--RGB_Hist', action='store', help='RGB: Number of bins for histogram, default 16', type=int, default=16)
parser.add_argument('--RGB_CI', action='store', help='RGB: Max Color Intensity [0 to VALUE], default 256', type=int, default=256)
parser.add_argument('--RGB_NMinMax', action='store_true', help='RGB: Use option to actvate MinMax Norm, default distribtion')

parser.add_argument('--HSV', action='store_true', help='Use option to activate HSV')
parser.add_argument('--HSV_H', action='store', help='HSV: Number of bins for Hue, default 8', type=int, default=8)
parser.add_argument('--HSV_S', action='store', help='HSV: Number of bins for Saturation, default 3', type=int, default=3)
parser.add_argument('--HSV_V', action='store', help='HSV: Number of bins for Value, default 3', type=int, default=3)
parser.add_argument('--HSV_NMinMax', action='store_true', help='HSV: Use option to actvate MinMax Norm, default distribtion')

parser.add_argument('--SIFT', action='store_true', help='Use option to activate SIFT')
parser.add_argument('--SIFT_Cluster', action='store', help='SIFT: Number of k-means cluster, default 50', type=int, default=50)
parser.add_argument('--SIFT_NMinMax', action='store_true', help='SIFT: Use option to actvate MinMax Norm, default distribtion')
        
parser.add_argument('--SURF', action='store_true', help='Use option to activate SURF')
parser.add_argument('--SURF_Cluster', action='store', help='SURF: Number of k-means cluster, default 50', type=int, default=50)
parser.add_argument('--SURF_NMinMax', action='store_true', help='SURF: Use option to actvate MinMax Norm, default distribtion')

parser.add_argument('--HOG', action='store_true', help='Use option to activate HOG')
parser.add_argument('--HOG_CellD', action='store', help='HOG: CellDimension for local histograms, default 5', type=int, default=5)
parser.add_argument('--HOG_Orient', action='store', help='HOG: Number of bins of local histograms , default 8', type=int, default=8)
parser.add_argument('--HOG_Cluster', action='store', help='HOG: Number of k-means cluster, default 12', type=int, default=12)
parser.add_argument('--HOG_Iter', action='store', help='HOG: Max. number of iterations for clustering, default 100', type=int, default=100)


# CELL_DIMENSION is the dimension of the cells on which we will compute local histograms 
# NB_ORIENTATIONS is the number of bins of this local histograms 
# intuitively, if CELL_DIMENSION is small it's better to have a small NB_ORIENTATIONS in order to have meaningful local histograms
# NB_CLUSTERS is the number of bins of the global histograms (the number of clusters in the KMEANS algorithm used for the bag of word)
# MAXITER is the maximum number of iteration for the clustering algorithm

args = parser.parse_args()
path = args.path
NB_CORES = args.cores
nameDB = args.name

### Helper

# Function to transform the boolean deciscion of norm into a string
def boolNormToStr(norm):
        if(norm):
                return "MinMax"
        else:
                return "Distr"

### Main Programm

print "### Main Programm for Feature Extraction ###"
features = ""
if(args.RGB):
        features = features + "RGB "
if(args.HSV):
        features = features + "HSV "
if(args.SIFT):
        features = features + "SIFT "
if(args.SURF):
        features = features + "SURF "
if(args.HOG):
        features = features + "HOG"

print "Infos:\t NameDB=" + nameDB + ", Path=" + path + ", Cores=" + str(NB_CORES) + ", Features=" + features

################################ Read Images from Database
# Determine the Database to extract features

print "Start:\t Exportation of images from DB"

t_db_start = time.time()

# get dictionary to link classLabels Text to Integers
sClassLabels = DBCrawl.getClassLabels(path)

# Get all path from all images inclusive classLabel as Integer
dfImages,nameDB = DBCrawl.imgCrawl(path, sClassLabels, nameDB)

print "Done:\t Exportation of images from DB"

t_db  = t_db_start - time.time()

################################ Feature Extraction
print "Start:\t Features Extraction"

### Setup RGB
if(args.RGB):
        
        print "RGB:\t Start"
        t_rgb_start = time.time()
        
        numberOfBins = args.RGB_Hist
        maxColorIntensity = args.RGB_CI
        boolNormMinMax = args.RGB_NMinMax
        
        # Infos
        print "RGB:\t NumberOfBins=" + str(numberOfBins) + ", MaxColorIntensity=" + str(maxColorIntensity) + ", Norm=" + boolNormToStr(boolNormMinMax)
         
        # Extract Feature from DB
        rgb_feat_desc,rgb_f_extr_res = FeatExtraction.calcRGBColorHisto(nameDB, dfImages, numberOfBins, maxColorIntensity, boolNormMinMax)

        t_rgb = time.time() - t_rgb_start
        print "RGB:\t Done in: " + str(t_rgb) + "[s]"


### Setup HSV
if(args.HSV):
        print "HSV:\t Start"
        t_hsv_start = time.time()
        
        h_bins = args.HSV_H
        s_bins = args.HSV_S
        v_bins = args.HSV_V
        histSize = [h_bins, s_bins, v_bins]
        boolNormMinMax = args.HSV_NMinMax
        
        # Infos
        print "HSV:\t HSVBins=[" + str(h_bins) + "," + str(s_bins) + "," + str(v_bins) + "], Norm=" + boolNormToStr(boolNormMinMax)

        # Extract Feature from DB
        hsv_feat_desc,hsv_f_extr_res = FeatExtraction.calcHSVColorHisto(nameDB, dfImages, histSize, boolNormMinMax)
        t_hsv = time.time() - t_hsv_start
        print "HSV:\t Done in: " + str(t_hsv) + "[s]"



### Setup SIFT
if(args.SIFT):
        print "SIFT:\t Start"
        t_sift_start = time.time()
        
        boolSIFT = True
        cluster = args.SIFT_Cluster
        boolNormMinMax = args.SIFT_NMinMax
        
        print "SIFT:\t Cluster=" + str(cluster) + ", Norm=" + boolNormToStr(boolNormMinMax)

        sift_descriptors,sift_des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
        sift_feat_desc,sift_f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, sift_descriptors, sift_des_list, boolSIFT)
        t_sift = time.time() - t_sift_start 
        print "SIFT:\t Done in: " + str(t_sift) + "[s]"


### Setup SURF
if(args.SURF):
        print "SURF:\t Start"
        t_surf_start = time.time()
        
        boolSIFT = False
        cluster = args.SURF_Cluster
        boolNormMinMax = args.SURF_NMinMax
        
        print "SURF:\t Cluster=" + str(cluster) + ", Norm=" + boolNormToStr(boolNormMinMax)

        # Extract Feature from DB
        surf_descriptors,surf_des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
        surf_feat_desc,surf_f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, surf_descriptors, surf_des_list, boolSIFT)
        t_surf = time.time() - t_surf_start 
        print "SURF:\t Done in: " + str(t_surf) + "[s]"

### Setup HOG
if(args.HOG):
        print "HOG:\t Start"
        t_hog_start = time.time()
        
        CELL_DIMENSION = args.HOG_CellD
        NB_ORIENTATIONS = args.HOG_Orient
        NB_CLUSTERS = args.HOG_Cluster
        MAXITER = args.HOG_Iter
        
        print "HOG:\t CellDim=" + str(CELL_DIMENSION) + ", NbOrientations=" + str(NB_ORIENTATIONS) +", Cluster=" + str(NB_CLUSTERS) + ", MaxIter=" + str(MAXITER)

        # Extract Feature from DB
        hog_feat_desc,hog_f_extr_res = FeatExtraction.calcHOGParallel(nameDB, dfImages.values, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER, NB_CORES)
        #hog_feat_desc,hog_f_extr_res = FeatExtraction.calcHOG(nameDB, dfImages.values, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER)   
        t_hog = time.time() - t_hog_start
        print "HOG:\t Done in: " + str(t_hog) + "[s]"

print "Done:\t Features Extraction"


################################ SAVE TO FEATURES DATABASES
print "Start:\t Save Features to CSV Databases"

### Classlabels and Description
OutputfileNameClassLabels = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + nameDB + "-ClassLabels"
ExportResults.exportNumpyToCSV(dfImages.classLabel, OutputfileNameClassLabels, '%i')

fileNameClassLabels = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + nameDB + "-ClassLabels-Description"
ExportResults.exportPandasToCSV(sClassLabels, fileNameClassLabels)

format = '%1.30f'
### RGB
if(args.RGB):
        fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + rgb_feat_desc
        ExportResults.exportNumpyToCSV(rgb_f_extr_res, fileName, format)
        

### HSV
if(args.HSV):
        fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + hsv_feat_desc
        ExportResults.exportNumpyToCSV(hsv_f_extr_res, fileName, format)

### SIFT
if(args.SIFT):
        fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + sift_feat_desc
        ExportResults.exportNumpyToCSV(sift_f_extr_res, fileName, format)

### SURF
if(args.SURF):
        fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + surf_feat_desc
        ExportResults.exportNumpyToCSV(surf_f_extr_res, fileName, format)

### HOG
if(args.HOG):
        fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + hog_feat_desc
        ExportResults.exportNumpyToCSV(hog_f_extr_res, fileName, format)

print "Done:\t Save Features to CSV Databases"