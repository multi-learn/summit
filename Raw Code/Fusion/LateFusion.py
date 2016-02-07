#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np 
import sys

# Our method in multiclass classification will be One-vs-One or One-vs-All
# classifiers, so if we can get the output of these classifiers, we are 
# able to compute a score for each class in each mono-view classification


# decisions : (nbFeature * NB_CLASS) arrray with the OVO/OVA scores for each 
# 				feature and each class
# weights : (nbFeature) arrays with the weights for each feature
def weightedLinear(decisions, weights):  

	# Normalize weights ?
	# weights = weights/float(max(weights))

	fused = sum(np.array([featureScores * weight for weight,featureScores\
					 	in zip(weights, decisions)]))
	# print fused
	
	return np.argmax(fused)


# For majority voting, we have a problem : we have 5 fetures and 101 classes 
# on Calthech, so if each feature votes for one class, we can't find a good 
# result
def majorityVoting(decisions, NB_CLASS):
	votes = np.zeros(NB_CLASS)
	nbFeature = len(decisions)

	for featureClassification in decisions:
		votes[featureClassification]+=1
	nbMaximum = len(np.where(votes==max(votes))[0])

	try:
		assert nbMaximum != nbFeature
	except:
		print "Majority voting can't decide, each classifier has voted for a different class"
		raise 

# Can be upgraded by restarting a new classification process if 
# there are multiple maximums : 
# 	while nbMaximum>1:
# 		relearn with only the classes that have a maximum number of vote
# 		votes = revote
# 		nbMaximum = len(np.where(votes==max(votes))[0])
	return np.argmax(votes)
	

# Main for testing
if __name__ == '__main__':
	nbFeature = 5
	NB_CLASS = 12
	TRUE_CLASS = 3

	decisionsEasy = np.array([np.zeros(NB_CLASS) for i in range(nbFeature)])
	for decision in decisionsEasy:
		decision[TRUE_CLASS]=12
	# print decisionsEasy

	decisionsHard = np.array([np.zeros(NB_CLASS) for i in range(nbFeature)])
	for decision in decisionsHard:
		decision[TRUE_CLASS]=12
	decisionsHard[nbFeature-2]=np.zeros(NB_CLASS)+1400
	decisionsHard[nbFeature-2][TRUE_CLASS]-=110

	decisionsMajority = np.array([TRUE_CLASS,TRUE_CLASS,TRUE_CLASS,1,5])
	decisionsMajorityFail = np.array([1,2,3,4,5])

	weights = np.random.rand(nbFeature)
	weights[nbFeature-2] = 2


	print weightedLinear(decisionsEasy, weights)
	print weightedLinear(decisionsHard, weights)
	print majorityVoting(decisionsMajority, NB_CLASS)
	print majorityVoting(decisionsMajorityFail, NB_CLASS)