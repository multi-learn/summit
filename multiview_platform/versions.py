# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def test_versions():
    """Used to test if all prerequisites are installed"""
    is_up_to_date = True
    to_install = []

    # try:
    #     import sys
    # except ImportError:
    #     raise
    #
    # try:
    #     import cvxopt
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("cvxopt")
    #
    # try:
    #     import pyscm
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("pyscm")
    #
    # try:
    #     import numpy
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("numpy")
    #
    # try:
    #     import scipy
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("scipy")
    #
    # try:
    #     import matplotlib
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("matplotlib")
    #
    # try:
    #     import sklearn
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("sklearn")
    #
    # try:
    #     import logging
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("logging")
    #
    # try:
    #     import joblib
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("joblib")
    #
    # try:
    #     import argparse
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("argparse")
    #
    # try:
    #     import h5py  #
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("h5py")
    #
    # # try:
    # #     import graphviz  #
    # # except ImportError:
    # #     is_up_to_date = False
    # #     to_install.append("graphviz")
    #
    # try:
    #     import pickle  #
    # except ImportError:
    #     is_up_to_date = False
    #     to_install.append("pickle")
    #
    # if not is_up_to_date:
    #     print(
    #         "You can't run at the moment, please install the following modules : \n" + "\n".join(
    #             to_install))
    #     quit()


if __name__ == "__main__":
    test_versions()
