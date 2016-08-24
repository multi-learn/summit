from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import StratifiedShuffleSplit as split
import numpy as np
# from sklearn.multiclass import OneVsRestClassifier
from ModifiedMulticlass import OneVsRestClassifier

# Add weights


def Kover(data, labels, arg, weights,):
    isBad = False
    subSamplingRatio = arg[0]

    return classifier, prediction, isBad