# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def testVersions():
    """Used to test if all prerequisites are installed"""
    isUpToDate = True
    toInstall = []

    try:
        import sys
    except ImportError:
        raise

    try:
        import pyscm
    except ImportError:
        isUpToDate = False
        toInstall.append("pyscm")

    try:
        import numpy
    except ImportError:
        isUpToDate = False
        toInstall.append("numpy")

    try:
        import scipy
    except ImportError:
        isUpToDate = False
        toInstall.append("scipy")

    try:
        import matplotlib
    except ImportError:
        isUpToDate = False
        toInstall.append("matplotlib")

    try:
        import sklearn
    except ImportError:
        isUpToDate = False
        toInstall.append("sklearn")

    try:
        import logging
    except ImportError:
        isUpToDate = False
        toInstall.append("logging")

    try:
        import joblib
    except ImportError:
        isUpToDate = False
        toInstall.append("joblib")

    try:
        import argparse
    except ImportError:
        isUpToDate = False
        toInstall.append("argparse")

    try:
        import h5py  #
    except ImportError:
        isUpToDate = False
        toInstall.append("h5py")

    # try:
    #     import graphviz  #
    # except ImportError:
    #     isUpToDate = False
    #     toInstall.append("graphviz")

    try:
        import pickle  #
    except ImportError:
        isUpToDate = False
        toInstall.append("pickle")

    if not isUpToDate:
        print("You can't run at the moment, please install the following modules : \n"+ "\n".join(toInstall))
        quit()

if __name__== "__main__":
    testVersions()