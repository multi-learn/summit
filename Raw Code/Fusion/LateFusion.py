#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier


# Our method in multiclass classification will be One-vs-One or One-vs-All
# classifiers, so if we can get the output of these classifiers, we are 
# able to compute a score for each class in each mono-view classification


# monoViewDecisions : (nbExample * nbFeature * NB_CLASS) array with the OVO/OVA scores for each 
# 				example, feature and each class



# weights : (nbFeature) array with the weights for each feature
def weightedLinear(monoViewDecisions, weights):
    # Normalize weights ?
    # weights = weights/float(max(weights))

    fusedExamples = np.array([sum(np.array([featureScores * weight for weight, featureScores \
                                            in zip(weights, exampleDecisions)])) for exampleDecisions in
                              monoViewDecisions])
    # print fused

    return np.array([np.argmax(fusedExample) for fusedExample in fusedExamples])


# The SVMClassifier is here used to find the right weights for linear fusion
# Here we have a function to train it, one to fuse. 
# And one to do both.
def SVMForLinearFusionTrain(monoViewDecisions, labels):
    SVMClassifier = OneVsOneClassifier(SVC())
    SVMClassifier.fit(monoViewDecisions, labels)
    return SVMClassifier

def SVMForLinearFusionFuse(monoViewDecisions, SVMClassifier):
    labels = SVMClassifier.predict(monoViewDecisions)
    return labels

def SVMForLinearFusion(monoViewDecisions, labels):
    SVMClassifier = SVMForLinearFusionTrain(monoViewDecisions, labels)
    fusedLabels = SVMForLinearFusionFuse(monoViewDecisions, SVMClassifier)
    return fusedLabels


# For majority voting, we have a problem : we have 5 fetures and 101 classes
# on Calthech, so if each feature votes for one class, we can't find a good 
# result
def majorityVoting(monoViewDecisions, NB_CLASS):
    nbExample = len(monoViewDecisions)
    votes = np.array([np.zeros(NB_CLASS) for example in monoViewDecisions])
    for exampleIndice in range(nbExample):
        for featureClassification in monoViewDecisions[exampleIndice]:
            votes[exampleIndice, featureClassification] += 1
        nbMaximum = len(np.where(votes[exampleIndice] == max(votes[exampleIndice]))[0])
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



# For probabilistic classifiers, we need to add more late fusion methods
# For example, the bayesian inference
#           probabilisticClassifiers is a nbExample array of sklearn probabilistic classifiers
#                                    (such as Naive Bayesian Gaussian http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)
def bayesianInference(probabilisticClassifiers):
    nbFeatures = len(probabilisticClassifiers)
    classifiersProbasByFeature = np.array([probabilisticClassifier.class_prior_ \
                                        for probabilisticClassifier in probabilisticClassifiers])
    classifiersProbasByExample = np.transpose(classifiersProbasByFeature)
    probabilities = np.array([weightedProduct(featureProbas, weights) for featureProbas in classifiersProbas])
    return probabilities/sum(probabilities)     

def weightedProduct(featureProbas, weights):
    weightedProbas = pow(featureProbas, weights)
    product = np.prod(weightedProbas)
    return product


# Main for testing
if __name__ == '__main__':
    DATASET_LENGTH = 10
    nbFeature = 5
    NB_CLASS = 12
    TRUE_CLASS = 3
    LABELS = np.array([TRUE_CLASS for i in range(DATASET_LENGTH)])
    LABELS[0] = 0

    monoViewDecisionsEasy = np.array(
            [np.array([np.zeros(NB_CLASS) for i in range(nbFeature)]) for example in range(DATASET_LENGTH)])
    for exampleDecisions in monoViewDecisionsEasy:
        for decision in exampleDecisions:
            decision[TRUE_CLASS] = 12
    # print monoViewDecisionsEasy

    monoViewDecisionsHard = np.array(
            [np.array([np.zeros(NB_CLASS) for i in range(nbFeature)]) for example in range(DATASET_LENGTH)])
    for exampleDecisions in monoViewDecisionsHard:
        for decision in exampleDecisions:
            decision[TRUE_CLASS] = 12
        exampleDecisions[nbFeature - 2] = np.zeros(NB_CLASS) + 1400
        exampleDecisions[nbFeature - 2][TRUE_CLASS] -= 110

    monoViewDecisionsMajority = np.array(
            [np.array([TRUE_CLASS, TRUE_CLASS, TRUE_CLASS, 1, 5]) for example in range(DATASET_LENGTH)])
    monoViewDecisionsMajorityFail = np.array([np.array([1, 2, 3, 4, 5]) for example in range(DATASET_LENGTH)])

    weights = np.random.rand(nbFeature)
    weights[nbFeature - 2] = 2

    SVMClassifier = SVMForLinearFusionTrain(monoViewDecisionsMajority, LABELS)

    print weightedLinear(monoViewDecisionsEasy, weights)
    print weightedLinear(monoViewDecisionsHard, weights)
    print SVMForLinearFusionFuse(monoViewDecisionsMajority, SVMClassifier)
    print majorityVoting(monoViewDecisionsMajority, NB_CLASS)
    print majorityVoting(monoViewDecisionsMajorityFail, NB_CLASS)
