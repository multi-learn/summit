#!/usr/bin/env python

""" Script to render versions of modules used """

# Import built-in modules

# Import 3rd party modules

# Import own modules


# Author-Info
__author__ 	= "Nikolas Huelsmann"
__status__ 	= "Prototype"           # Production, Development, Prototype
__date__	= 2016-03-25

try:
    import sys
    print("Python-V.: " + sys.version)
except:
    print "Please install Python 2.7"

try:
    import cv2
    print("OpenCV2-V.: " + cv2.__version__)
except:
    print "Please install cv2 module"

try:
    import pandas as pd
    print("Pandas-V.: " + pd.__version__)
except:
    print "Please install pandas module"

try:
    import numpy
    print("Numpy-V.: " + numpy.version.version)
except:
    print "Please install numpy module"

try:
    import scipy
    print("Scipy-V.: " + scipy.__version__)
except:
    print "Please install scipy module"

try:
    import matplotlib
    print("Matplotlib-V.: " + matplotlib.__version__)
except:
    print "Please install matplotlib module"

try:
    import sklearn
    print("Sklearn-V.: " + sklearn.__version__)
except:
    print "Please install sklearn module"

try:
    import logging                          # To create Log-Files
    print("Logging: " + logging.__version__)
except:
    print "Please install logging module"

try:
    import joblib
    print("joblib: " + joblib.__version__)
except:
    print "Pease install joblib module"


