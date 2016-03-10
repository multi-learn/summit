#!/usr/bin/env python

""" Script to perform feature parameter optimisation """

# Import built-in modules
#import cv2                      # for OpenCV 
#import cv                       # for OpenCV
#import datetime                 # for TimeStamp in CSVFile
#from scipy.cluster.vq import *  # for Clustering http://docs.scipy.org/doc/scipy/reference/cluster.vq.html
#import numpy as np              # for arrays
#import time                     # for time calculations
from argparse import ArgumentParser # for acommand line arguments

# Import 3rd party modules

# Import own modules
#import DBCrawl			# Functions to read Images from Database
#import ExportResults            # Functions to render results
#import FeatExtraction           # Functions to extract the features from Database   

# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Development" #Production, Development, Prototype
__date__	= 2016-03-10

### Argument Parser

parser = ArgumentParser(description='Perform feature parameter optimisation')

parser.add_argument('-p', '--path', action='store', help='Path to the database', default='D:\\CaltechMini')
parser.add_argument('-c', '--cores', action='store', type=int, help='Nb cores used for parallelization', default=1)

args = parser.parse_args()