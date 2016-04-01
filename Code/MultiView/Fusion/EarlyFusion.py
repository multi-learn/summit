#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np


# TODO :
# Linear Weighted Fusion done E&L
# Bayesian Inference  /!\ Statistically independant => ? Patern Classification done L
# SVM ? done L
# Dempster Schafer Theory  Need to create our own dempster schaffer function
# Dynamic Bayesian Networks ? -> More useful for many features
# Neural Network ???
# Maximum Entropy Model ? -> Need to use a max entropy classifier to fuse it
# Kalman Filter ?
# Particle Filter ?

def linearWeightedFusion(toFuse, weights):
    # Normalize weights ?
    # weights = weights/float(max(weights))

    weighted = np.array(
            [np.array([feature * weights for (feature, weight) in zip(exampleToFuse, weights)]).flatten() for
             exampleToFuse in toFuse])
    return weighted


if __name__ == '__main__':
    pass
