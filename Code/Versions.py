#!/usr/bin/env python

""" Script to render versions of modules used """

# Import built-in modules

# Import 3rd party modules

# Import own modules


# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype
# __date__ = 2016 - 03 - 25


def testVersions():
    isUpToDate = True
    toInstall = []
    try:
        import sys
        # print("Python-V.: " + sys.version)
    except:
        # print "Please install Python 2.7"
        raise

    try:
        import pyscm
    except:
        # print "Please install pyscm"
        isUpToDate = False
        toInstall.append("pyscm")

    try:
        import numpy
        # print("Numpy-V.: " + numpy.version.version)
    except:
        # print "Please install numpy module"
        isUpToDate = False
        toInstall.append("numpy")

    try:
        import scipy
        # print("Scipy-V.: " + scipy.__version__)
    except:
        # print "Please install scipy module"
        isUpToDate = False
        toInstall.append("scipy")

    try:
        import matplotlib
        # print("Matplotlib-V.: " + matplotlib.__version__)
    except:
        # print "Please install matplotlib module"
        isUpToDate = False
        toInstall.append("matplotlib")

    try:
        import sklearn
        # print("Sklearn-V.: " + sklearn.__version__)
    except:
        # print "Please install sklearn module"
        isUpToDate = False
        toInstall.append("sklearn")

    try:
        import logging  # To create Log-Files
        # print("Logging: " + logging.__version__)
    except:
        # print "Please install logging module"
        isUpToDate = False
        toInstall.append("logging")

    try:
        import joblib
        # print("joblib: " + joblib.__version__)
    except:
        # print "Pease install joblib module"
        isUpToDate = False
        toInstall.append("joblib")

    try:
        import argparse
        # print("argparse: " + argparse.__version__)
    except:
        # print "Pease install argparse module"
        isUpToDate = False
        toInstall.append("argparse")

    try:
        import h5py  #
        # print("h5py: " + h5py.__version__)
    except:
        # print "Pease install h5py module"
        isUpToDate = False
        toInstall.append("h5py")

    try:
        import graphviz  #
    except:
        isUpToDate = False
        toInstall.append("graphviz")

    try:
        import pickle  #
    except:
        isUpToDate = False
        toInstall.append("pickle")

    if not isUpToDate:
        print("You can't run at the moment, please install the following modules : \n"+ "\n".join(toInstall))
        quit()

if __name__== "__main__":
    testVersions()