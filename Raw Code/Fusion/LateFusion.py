#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np 
import sys
from sklearn.svm import SVC

# Our method in multiclass classification will be One-vs-One or One-vs-All
# classifiers, so if we can get the output of these classifiers, we are 
# able to compute a score for each class in each mono-view classification


# decisions : (nbExample * nbFeature * NB_CLASS) array with the OVO/OVA scores for each 
# 				example, feature and each class


# weights : (nbFeature) array with the weights for each feature
def weightedLinear(decisions, weights):  

	# Normalize weights ?
	# weights = weights/float(max(weights))

	fusedExamples = np.array([sum(np.array([featureScores * weight for weight,featureScores\
					 	in zip(weights, exampleDecisions)])) for exampleDecisions in decisions])
	# print fused
	
	return np.array([np.argmax(fusedExample) for fusedExample in fusedExamples])


# The SVMClassifier is here used to find the right weights for linearfusion
def SVMForLinearFusionTrain(decisions, labels):
	SVMClassifier = SVC()
	SVMClassifier.fit(decisions, labels)
	return SVMClassifier

def SVMForLinearFusionFuse(decisions, SVMClassifier):
	labels = SVMClassifier.predict(decisions)
	return labels



# For majority voting, we have a problem : we have 5 fetures and 101 classes 
# on Calthech, so if each feature votes for one class, we can't find a good 
# result
def majorityVoting(decisions, NB_CLASS):
	nbExample = len(decisions)
	votes = np.array([np.zeros(NB_CLASS) for example in decisions])
	for exampleIndice in range(nbExample):
		for featureClassification in decisions[exampleIndice]:
			votes[exampleIndice, featureClassification]+=1
		nbMaximum = len(np.where(votes[exampleIndice]==max(votes[exampleIndice]))[0])
		try:
			assert nbMaximum != nbFeature
		except:
			print "Majority voting can't decide, each classifier has voted for a different class"
			raise 

# Can be upgraded by restarting a new classification process if 
# there are multiple maximums ?: 
# 	while nbMaximum>1:
# 		relearn with only the classes that have a maximum number of vote
# 		votes = revote
# 		nbMaximum = len(np.where(votes==max(votes))[0])

	return np.array([np.argmax(exampleVotes) for exampleVotes in votes])




# Main for testing
if __name__ == '__main__':
	DATASET_LENGTH = 10
	nbFeature = 5
	NB_CLASS = 12
	TRUE_CLASS = 3
	LABELS = np.array([TRUE_CLASS for i in range(DATASET_LENGTH)])
	LABELS[0] = 0

	decisionsEasy = np.array([np.array([np.zeros(NB_CLASS) for i in range(nbFeature)])for example in range(DATASET_LENGTH)])
	for exampleDecisions in decisionsEasy:
		for decision in exampleDecisions:
			decision[TRUE_CLASS]=12
	# print decisionsEasy

	decisionsHard = np.array([np.array([np.zeros(NB_CLASS) for i in range(nbFeature)])for example in range(DATASET_LENGTH)])
	for exampleDecisions in decisionsHard:
		for decision in exampleDecisions:
			decision[TRUE_CLASS]=12
		exampleDecisions[nbFeature-2]=np.zeros(NB_CLASS)+1400
		exampleDecisions[nbFeature-2][TRUE_CLASS]-=110

	decisionsMajority = np.array([np.array([TRUE_CLASS,TRUE_CLASS,TRUE_CLASS,1,5]) for example in range(DATASET_LENGTH)])
	decisionsMajorityFail = np.array([np.array([1,2,3,4,5]) for example in range(DATASET_LENGTH)])

	weights = np.random.rand(nbFeature)
	weights[nbFeature-2] = 2

	SVMClassifier = SVMForLinearFusionTrain(decisionsMajority, LABELS)


	print weightedLinear(decisionsEasy, weights)
	print weightedLinear(decisionsHard, weights)
	print SVMForLinearFusionFuse(decisionsMajority, SVMClassifier)
	print majorityVoting(decisionsMajority, NB_CLASS)
	print majorityVoting(decisionsMajorityFail, NB_CLASS)
