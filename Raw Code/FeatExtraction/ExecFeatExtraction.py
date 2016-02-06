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
__date__	= 2016-02-04

### Argument Parser

parser = ArgumentParser(description='Perform feature parameter optimisation')

parser.add_argument('-p', '--path', action='store', help='Path to the database', default='D:\\CaltechMini')
parser.add_argument('-c', '--cores', action='store', type=int, help='Nb cores used for parallelization', default=1)

args = parser.parse_args()

path = args.path
NB_CORES = args.cores

### Main Programm

################################ Read Images from Database
# Determine the Database to extract features

print "### Main Programm for Feature Extraction ###"
# path ="D:\\CaltechMini"
path = args.path
nameDB = "CT-Mini"

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
t_rgb_start = time.time()
print "RGB:\t Start"

numberOfBins = 16
maxColorIntensity = 256
boolNormMinMax = False
 
# Extract Feature from DB
rgb_feat_desc,rgb_f_extr_res = FeatExtraction.calcRGBColorHisto(nameDB, dfImages, numberOfBins, maxColorIntensity, boolNormMinMax)

t_rgb = time.time() - t_rgb_start
print "RGB:\t Done in: " + str(t_rgb) + "[s]"


### Setup HSV
t_hsv_start = time.time()
print "HSV:\t Start"
h_bins = 8 
s_bins = 3
v_bins = 3
histSize = [h_bins, s_bins, v_bins]
boolNormMinMax = False

# Extract Feature from DB
hsv_feat_desc,hsv_f_extr_res = FeatExtraction.calcHSVColorHisto(nameDB, dfImages, histSize, boolNormMinMax)
t_hsv = time.time() - t_hsv_start
print "HSV:\t Done in: " + str(t_hsv) + "[s]"



### Setup SIFT
t_sift_start = time.time()
print "SIFT:\t Start"
boolSIFT = True
cluster = 50
boolNormMinMax = False

sift_descriptors,sift_des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
sift_feat_desc,sift_f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, sift_descriptors, sift_des_list, boolSIFT)
t_sift = time.time() - t_sift_start 
print "SIFT:\t Done in: " + str(t_sift) + "[s]"


### Setup SURF
t_surf_start = time.time()
print "SURF:\t Start"
boolSIFT = False
cluster = 50
boolNormMinMax = False

# Extract Feature from DB
surf_descriptors,surf_des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
surf_feat_desc,surf_f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, surf_descriptors, surf_des_list, boolSIFT)
t_surf = time.time() - t_surf_start 
print "SURF:\t Done in: " + str(t_surf) + "[s]"

### Setup HOG
t_hog_start = time.time()
print "HOG:\t Start"
CELL_DIMENSION = 5
NB_ORIENTATIONS = 8
NB_CLUSTERS = 12
MAXITER = 100
NB_CORES = 1

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
ExportResults.exportNumpyToCSV(dfImages.classLabel, OutputfileNameClassLabels)

fileNameClassLabels = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + nameDB + "-ClassLabels-Description"
ExportResults.exportPandasToCSV(sClassLabels, fileNameClassLabels)

### RGB
fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + rgb_feat_desc
ExportResults.exportNumpyToCSV(rgb_f_extr_res, fileName)

### HSV
fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + hsv_feat_desc
ExportResults.exportNumpyToCSV(hsv_f_extr_res, fileName)

### SIFT
fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + sift_feat_desc
ExportResults.exportNumpyToCSV(sift_f_extr_res, fileName)

### SURF
fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + surf_feat_desc
ExportResults.exportNumpyToCSV(surf_f_extr_res, fileName)

### HOG
fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + hog_feat_desc
ExportResults.exportNumpyToCSV(hog_f_extr_res, fileName)

print "Done:\t Save Features to CSV Databases"