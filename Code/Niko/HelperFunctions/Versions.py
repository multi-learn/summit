#!/usr/bin/env python

""" Script to render versions of modules used """

# Import built-in modules

# Import 3rd party modules

# Import own modules


# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25

import sys
print("Python-V.: " + sys.version)

import cv2
print("OpenCV2-V.: " + cv2.__version__)

import pandas as pd
print("Pandas-V.: " + pd.__version__)

import numpy
print("Numpy-V.: " + numpy.version.version)

import scipy
print("Scipy-V.: " + scipy.__version__)

import matplotlib
print("Matplotlib-V.: " + matplotlib.__version__)

import sklearn
print("Sklearn-V.: " + sklearn.__version__)


import logging                          # To create Log-Files  
print("Logging: " + logging.__version__)

