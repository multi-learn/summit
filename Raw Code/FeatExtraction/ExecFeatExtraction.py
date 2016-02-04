#!/usr/bin/env python

""" Script to perform feature parameter optimisation """

# Import built-in modules
import cv2                      # for OpenCV 
import cv                       # for OpenCV
import datetime                 # for TimeStamp in CSVFile
from scipy.cluster.vq import *  # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
import numpy as np              # for arrays

# Import 3rd party modules

# Import own modules
import DBCrawl			# Functions to read Images from Database
import ExportResults            # Functions to render results
import FeatExtraction           # Functions to extract the features from Database   

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-02-04

### Main Programm

################################ Read Images from Database
# Determine the Database to extract features

print "### Main Programm for Feature Extraction ###"
path ="D:\\CaltechMini"
nameDB = "CT-Mini"

print "Start:\t Exportation of images from DB"

# get dictionary to link classLabels Text to Integers
sClassLabels = DBCrawl.getClassLabels(path)

# Get all path from all images inclusive classLabel as Integer
dfImages,nameDB = DBCrawl.imgCrawl(path, sClassLabels, nameDB)

print "Done:\t Exportation of images from DB"


################################ Feature Extraction
print "Start:\t Features Extraction"

### Setup RGB
print "RGB:\t Start"
numberOfBins = 16
maxColorIntensity = 256
boolNormMinMax = False
 
# Extract Feature from DB
rgb_feat_desc,rgb_f_extr_res = FeatExtraction.calcRGBColorHisto(nameDB, dfImages, numberOfBins, maxColorIntensity, boolNormMinMax)
print "RGB:\t Done"


### Setup HSV
print "HSV:\t Start"
h_bins = 8 
s_bins = 3
v_bins = 3
histSize = [h_bins, s_bins, v_bins]
boolNormMinMax = False

# Extract Feature from DB
hsv_feat_desc,hsv_f_extr_res = FeatExtraction.calcHSVColorHisto(nameDB, dfImages, histSize, boolNormMinMax)
print "HSV:\t Done"


### Setup SIFT
print "SIFT:\t Start"
boolSIFT = True
cluster = 50
boolNormMinMax = False

sift_descriptors,sift_des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
sift_feat_desc,sift_f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, sift_descriptors, sift_des_list, boolSIFT)
print "SIFT:\t Done"


### Setup SURF
print "SURF:\t Start"
boolSIFT = False
cluster = 50
boolNormMinMax = False

# Extract Feature from DB
surf_descriptors,surf_des_list = FeatExtraction.calcSURFSIFTDescriptors(dfImages, boolSIFT)
surf_feat_desc,surf_f_extr_res = FeatExtraction.calcSURFSIFTHisto(nameDB, dfImages, cluster, boolNormMinMax, surf_descriptors, surf_des_list, boolSIFT)
print "SURF:\t Done"


### Setup HOG
#print "HOG:\t Start"
#CELL_DIMENSION = 5
#NB_ORIENTATIONS = 8
#NB_CLUSTERS = 12
#MAXITER = 100

# Extract Feature from DB
#hog_feat_desc,hof_f_extr_res = FeatExtraction.calcHOGParallel(nameDB, npImages, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER, NB_CORES)
#hog_feat_desc,hog_f_extr_res = FeatExtraction.calcHOG(nameDB, dfImages, CELL_DIMENSION, NB_ORIENTATIONS, NB_CLUSTERS, MAXITER)   
#print "HOG:\t Done"

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
#fileName = datetime.datetime.now().strftime("%Y_%m_%d") + "-" + hog_feat_desc
#ExportResults.exportNumpyToCSV(hog_f_extr_res, fileName)

print "Done:\t Save Features to CSV Databases"