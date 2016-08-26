#!/usr/bin/env python
# -*- encoding: utf-8

import numpy as np
from joblib import Parallel, delayed
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

import MonoviewClassifiers


# Our method in multiclass classification will be One-vs-One or One-vs-All
# classifiers, so if we can get the output of these classifiers, we are 
# able to compute a score for each class in each mono-view classification


# monoViewDecisions : (nbExample * nbFeature * NB_CLASS) array with the OVO/OVA scores for each 
# 				example, feature and each class
# weights : (nbFeature) array with the weights for each feature
def fifMonoviewClassifier(classifierName, data, labels, classifierConfig):
    monoviewClassifier = getattr(MonoviewClassifiers, classifierName)
    classifier = monoviewClassifier.fit(data,labels,**dict((str(configIndex), config) for configIndex, config in
                                      enumerate(classifierConfig
                                                )))
    return classifier

class LateFusionClassifier(object):
    def __init__(self, monoviewClassifiersNames, monoviewClassifiersConfigs, NB_CORES=1):
        self.monoviewClassifiersNames = monoviewClassifiersNames
        self.monoviewClassifiersConfigs = monoviewClassifiersConfigs
        self.monoviewClassifiers = []
        self.nbCores = NB_CORES

    def fit_hdf5(self, DATASET, trainIndices=None):
        if trainIndices == None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        nbView = DATASET.get("Metadata").attrs["nbView"]
        self.monoviewClassifiers = Parallel(n_jobs=self.nbCores)(
            delayed(fifMonoviewClassifier)(self.monoviewClassifiersNames[viewIndex],
                                              DATASET.get("View"+str(viewIndex))[trainIndices, :],
                                              DATASET.get("labels")[trainIndices],
                                              self.monoviewClassifiersConfigs[viewIndex])
            for viewIndex in range(nbView))


class WeightedLinear(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['monoviewClassifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.weights = map(float, kwargs['fusionMethodConfig'][0])

    def predict_hdf5(self, DATASET, usedIndices=None):
        # Normalize weights ?
        # weights = weights/float(max(weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            predictedLabels = []
            viewScores = np.zeros((DATASET.get("Metadata").attrs["nbView"], len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
            for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
                viewScores[viewIndex] = self.monoviewClassifiers[viewIndex].predict_proba(
                    DATASET.get("View" + str(viewIndex))[usedIndices])
            for currentIndex, usedIndex in enumerate(usedIndices):
                predictedLabel = np.argmax(np.array(
                    [max(viewScore) * weight for viewScore, weight in zip(viewScores[:, currentIndex], self.weights)],
                    dtype=float))
                predictedLabels.append(predictedLabel)
                # fusedExamples = np.array([sum(np.array([featureScores * weight for weight, featureScores in zip(weights, exampleDecisions)])) for exampleDecisions in monoViewDecisions])
        else:
            predictedLabels = []

        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Weighted linear using a weight for each view : "+", ".join(self.weights) + \
                       "\n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString



# The SVMClassifier is here used to find the right weights for linear fusion
# Here we have a function to train it, one to fuse. 
# And one to do both.
class SVMForLinear(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['monoviewClassifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.SVMClassifier = None

    def fit_hdf5(self, DATASET, trainIndices=None):
        if trainIndices == None:
            trainIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        nbViews = DATASET.get("Metadata").attrs["nbView"]
        for viewIndex in range(nbViews):
            monoviewClassifier = getattr(MonoviewClassifiers, self.monoviewClassifiersNames[viewIndex])
            self.monoviewClassifiers.append(
                monoviewClassifier.fit(DATASET.get("View" + str(viewIndex))[trainIndices],
                                       DATASET.get("labels")[trainIndices],
                                       NB_CORES=self.nbCores,
                                       **dict((str(configIndex), config) for configIndex, config in
                                              enumerate(self.monoviewClassifiersConfigs[viewIndex]
                                                        ))))
        self.SVMForLinearFusionFit(DATASET, usedIndices=trainIndices)

    def predict_hdf5(self, DATASET, usedIndices=None):
        # Normalize weights ?
        # weights = weights/float(max(weights))
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            monoviewDecisions = np.zeros((len(usedIndices), DATASET.get("Metadata").attrs["nbView"]), dtype=int)
            for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
                monoviewClassifier = getattr(MonoviewClassifiers, self.monoviewClassifiersNames[viewIndex])
                monoviewDecisions[:, viewIndex] = self.monoviewClassifiers[viewIndex].predict(
                    DATASET.get("View" + str(viewIndex))[usedIndices])
            predictedLabels = self.SVMClassifier.predict(monoviewDecisions)
        else:
            predictedLabels = []
        return predictedLabels

    def SVMForLinearFusionFit(self, DATASET, usedIndices=None):
        self.SVMClassifier = OneVsOneClassifier(SVC())
        monoViewDecisions = np.zeros((len(usedIndices), DATASET.get("Metadata").attrs["nbView"]), dtype=int)
        for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
            monoViewDecisions[:, viewIndex] = self.monoviewClassifiers[viewIndex].predict(
                DATASET.get("View" + str(viewIndex))[usedIndices])

        self.SVMClassifier.fit(monoViewDecisions, DATASET.get("labels")[usedIndices])

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with SVM for linear \n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString


# For majority voting, we have a problem : we have 5 fetures and 101 classes
# on Calthech, so if each feature votes for one class, we can't find a good 
# result
class MajorityVoting(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['monoviewClassifiersConfigs'],
                                      NB_CORES=NB_CORES)

    def predict_hdf5(self, DATASET, usedIndices=None):
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if usedIndices:
            datasetLength = len(usedIndices)
            votes = np.zeros((datasetLength, DATASET.get("Metadata").attrs["nbClass"]), dtype=int)
            monoViewDecisions = np.zeros((len(usedIndices),DATASET.get("Metadata").attrs["nbView"]), dtype=int)
            for viewIndex in range(DATASET.get("Metadata").attrs["nbView"]):
                monoViewDecisions[:, viewIndex] = self.monoviewClassifiers[viewIndex].predict(
                    DATASET.get("View" + str(viewIndex))[usedIndices])
            for exampleIndex in range(datasetLength):
                for featureClassification in monoViewDecisions[exampleIndex, :]:
                    votes[exampleIndex, featureClassification] += 1
                nbMaximum = len(np.where(votes[exampleIndex] == max(votes[exampleIndex]))[0])
                try:
                    assert nbMaximum != DATASET.get("Metadata").attrs["nbView"]
                except:
                    print "Majority voting can't decide, each classifier has voted for a different class"
                    raise
            predictedLabels = np.argmax(votes, axis=1)
            # Can be upgraded by restarting a new classification process if
            # there are multiple maximums ?:
            # 	while nbMaximum>1:
            # 		relearn with only the classes that have a maximum number of vote
            # 		votes = revote
            # 		nbMaximum = len(np.where(votes==max(votes))[0])
        else:
            predictedLabels = []
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Majority Voting \n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString


# For probabilistic classifiers, we need to add more late fusion methods
# For example, in the bayesian inference
# probabilisticClassifiers is a nbExample array of sklearn probabilistic classifiers
# (such as Naive Bayesian Gaussian
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)
class BayesianInference(LateFusionClassifier):
    def __init__(self, NB_CORES=1, **kwargs):
        LateFusionClassifier.__init__(self, kwargs['classifiersNames'], kwargs['monoviewClassifiersConfigs'],
                                      NB_CORES=NB_CORES)
        self.weights = np.array(map(float, kwargs['fusionMethodConfig'][0]))

    def predict_hdf5(self, DATASET, usedIndices=None):
        nbView = DATASET.get("nbView").value
        if usedIndices == None:
            usedIndices = range(DATASET.get("Metadata").attrs["datasetLength"])
        if sum(self.weights)!=1.0:
            self.weights = self.weights/sum(self.weights)
        if usedIndices:

            viewScores = np.zeros((nbView, len(usedIndices), DATASET.get("Metadata").attrs["nbClass"]))
            for viewIndex in range(nbView):
                viewScores[viewIndex] = np.power(self.monoviewClassifiers[viewIndex].predict_proba(DATASET.get("View" + str(viewIndex))
                                                                                          [usedIndices]),
                                                 self.weights[viewIndex])
            predictedLabels = np.argmax(np.prod(viewScores, axis=1), axis=1)
        else:
            predictedLabels = []
        return predictedLabels

    def getConfig(self, fusionMethodConfig, monoviewClassifiersNames,monoviewClassifiersConfigs):
        configString = "with Bayesian Inference using a weight for each view : "+", ".join(self.weights) + \
                       "\n\t-With monoview classifiers : "
        for monoviewClassifierConfig, monoviewClassifierName in zip(monoviewClassifiersConfigs, monoviewClassifiersNames):
            monoviewClassifierModule = getattr(MonoviewClassifiers, monoviewClassifierName)
            configString += monoviewClassifierModule.getConfig(monoviewClassifierConfig)
        return configString




#
# def weightedProduct(featureProbas, weights):
#     try:
#         assert np.sum(weights) == 1.0
#     except:
#         print "Need to give a weight array that sums to one"
#         raise
#     weightedProbas = np.power(featureProbas, weights)
#     product = np.prod(weightedProbas)
#     return product

#
#
# # Main for testing
# if __name__ == '__main__':
#     DATASET_LENGTH = 10
#     nbFeature = 5
#     NB_CLASS = 12
#     TRUE_CLASS = 3
#     LABELS = np.array([TRUE_CLASS for i in range(DATASET_LENGTH)])
#     LABELS[0] = 0
#
#     monoViewDecisionsEasy = np.array(
#             [np.array([np.zeros(NB_CLASS) for i in range(nbFeature)]) for example in range(DATASET_LENGTH)])
#     for exampleDecisions in monoViewDecisionsEasy:
#         for decision in exampleDecisions:
#             decision[TRUE_CLASS] = 12
#     # print monoViewDecisionsEasy
#
#     monoViewDecisionsHard = np.array(
#             [np.array([np.zeros(NB_CLASS) for i in range(nbFeature)]) for example in range(DATASET_LENGTH)])
#     for exampleDecisions in monoViewDecisionsHard:
#         for decision in exampleDecisions:
#             decision[TRUE_CLASS] = 12
#         exampleDecisions[nbFeature - 2] = np.zeros(NB_CLASS) + 1400
#         exampleDecisions[nbFeature - 2][TRUE_CLASS] -= 110
#
#     monoViewDecisionsMajority = np.array(
#             [np.array([TRUE_CLASS, TRUE_CLASS, TRUE_CLASS, 1, 5]) for example in range(DATASET_LENGTH)])
#     monoViewDecisionsMajorityFail = np.array([np.array([1, 2, 3, 4, 5]) for example in range(DATASET_LENGTH)])
#
#     weights = np.random.rand(nbFeature)
#     weights[nbFeature - 2] = 2
#
#     SVMClassifier = SVMForLinearFusionTrain(monoViewDecisionsMajority, LABELS)
#
#     print weightedLinear(monoViewDecisionsEasy, weights)
#     print weightedLinear(monoViewDecisionsHard, weights)
#     print SVMForLinearFusionFuse(monoViewDecisionsMajority, SVMClassifier)
#     print majorityVoting(monoViewDecisionsMajority, NB_CLASS)
#     print majorityVoting(monoViewDecisionsMajorityFail, NB_CLASS)