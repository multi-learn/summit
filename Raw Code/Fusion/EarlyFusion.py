#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np

# TODO :
# Linear Weighted Fusion
# Bayesian Inference  /!\ Statistically independant => ? 
# Dempster Schafer Theory  /!\ Inunderstandable
# Dynamic Bayesian Networks ?
# Neural Network ?
# Maximum Entropy Model ?
# Kalman Filter ?
# Particle Filter ?

def linearWeightedFusion (toFuse, weights):
	
	# Normalize weights ?
	# weights = weights/float(max(weights))

	weighted = np.array([feature*weights for (feature, weight) in zip(toFuse, weights)])
	fused = weighted.flatten()

	return fused


