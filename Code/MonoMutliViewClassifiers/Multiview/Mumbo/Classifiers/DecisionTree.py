import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from ModifiedMulticlass import OneVsRestClassifier
from SubSampling import subSample
import logging

# Add weights

from .... import Metrics


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=10, criterion="gini", splitter="best", subSampling=1.0, randomState=None, **kwargs):
        if kwargs:
            self.depth = kwargs["depth"]
            self.criterion = kwargs["criterion"]
            self.splitter = kwargs["splitter"]
            self.subSampling = kwargs["subSampling"]
            self.randomState = kwargs["randomState"]
        else:
            self.depth = depth
            self.criterion = criterion
            self.splitter = splitter
            self.subSampling = subSampling
            if randomState is None:
                self.randomState=np.random.RandomState()
            else:
                self.randomState=randomState
        self.decisionTree = sklearn.tree.DecisionTreeClassifier(splitter=self.splitter, criterion=self.criterion, max_depth=self.depth)

    def fit(self, data, labels, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(data))/len(data)

        if self.subSampling != 1.0:
            subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, self.subSampling, self.randomState,
                                                                            weights=sample_weight)
        else:
            subSampledData, subSampledLabels, subSampledWeights = data, labels, sample_weight

        self.decisionTree.fit(subSampledData, subSampledLabels, sample_weight=subSampledWeights)

        return self

    def fit_hdf5(self, data, labels, weights, metric):
        metricModule = getattr(Metrics, metric[0])
        if metric[1] is not None:
            metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
        else:
            metricKWARGS = {}
        if weights is None:
            weights = np.ones(len(data))/len(data)

        # Check that X and y have correct shape
        if self.subSampling != 1.0:
            subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, self.subSampling, self.randomState,
                                                                            weights=weights)
        else:
            subSampledData, subSampledLabels, subSampledWeights = data, labels, weights
        # self.subSampledData = subSampledData
        # self.
        # self.
        # Store the classes seen during fit
        self.decisionTree.fit(subSampledData, subSampledLabels, sample_weight=subSampledWeights)
        prediction = self.decisionTree.predict(data)
        metricKWARGS = {"0":weights}
        averageScore = metricModule.score(labels, prediction, **metricKWARGS)
        if averageScore < 0.5:
            isBad = True
        else:
            isBad = False

        # self.X_ = X
        # self.y_ = y
        # Return the classifier
        # self.decisionTree, prediction, isBad, averageScore
        return self.decisionTree, prediction, isBad, averageScore

    def predict(self, data):

        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        # X = check_array(X)
        predictedLabels = self.decisionTree.predict(data)
        # closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return predictedLabels

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"depth": self.depth, "criterion": self.criterion, "splitter": self.splitter, "subSampling": self.subSampling}

    def set_params(self, **parameters):
        self.depth = parameters["depth"]
        self.criterion = parameters["criterion"]
        self.splitter = parameters["splitter"]
        self.subSampling = parameters["subSampling"]
        # for parameter, value in parameters.items():
        #     print parameter, value
        #     self.setattr(parameter, value)
        return self

# def DecisionTree(data, labels, arg, weights, randomState):
#     depth = int(arg[0])
#     subSampling = float(arg[1])
#     if subSampling != 1.0:
#         subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling, randomState,
#                                                                         weights=weights)
#     else:
#         subSampledData, subSampledLabels, subSampledWeights = data, labels, weights
#     isBad = False
#     classifier = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
#     # classifier = OneVsRestClassifier(tree.DecisionTreeClassifier(max_depth=depth))
#     classifier.fit(subSampledData, subSampledLabels, sample_weight=subSampledWeights)
#     prediction = classifier.predict(data)
#     accuracy = accuracy_score(labels, prediction)
#     if accuracy < 0.5:
#         isBad = True
#
#     return classifier, prediction, isBad, accuracy


def getKWARGS(argList, randomState):
    kwargs = {"depth":int(argList[0]), "criterion":argList[1], "splitter":argList[2], "subSampling":float(argList[3]), "randomState":randomState}
    return kwargs


def getConfig(classifierConfig):
    try:
        depth = classifierConfig["depth"]
        splitter = classifierConfig["splitter"]
        criterion = classifierConfig["criterion"]
        subSampling = classifierConfig["subSampling"]
        return 'with depth ' + str(depth) + ', ' + \
               'with splitter ' + splitter + ', ' + \
               'with criterion ' + criterion + ', ' + \
               ' sub-sampled at ' + str(subSampling) + ' '
    except KeyError:
        print classifierConfig



def findClosest(scores, base=0.5):
    diffToBase = 100.0
    bestSettingsIndex = 0
    for resultIndex, result in enumerate(scores):
        if abs(base - result) < diffToBase:
            diffToBase = abs(base - result)
            bestResult = result
            bestSettingsIndex = resultIndex
    return bestSettingsIndex


def hyperParamSearch(data, labels, randomState, metric=["accuracy_score", None], nbSubSamplingTests=20):
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    subSamplingRatios = np.arange(nbSubSamplingTests, dtype=float)/nbSubSamplingTests
    maxDepths = np.arange(1)+1
    criterions = ["gini", "entropy"]
    splitters = ["best", "random"]
    parameters = {"depth":maxDepths, "criterion":criterions, "splitter":splitters, "subSampling":subSamplingRatios}
    classifier = DecisionTree()
    grid = sklearn.model_selection.GridSearchCV(classifier, parameters, scoring=scorer)
    grid.fit(data, labels)
    GSSubSamplingRatios = grid.cv_results_["param_subSampling"]
    GSMaxDepths = grid.cv_results_["param_depth"]
    GSCriterions = grid.cv_results_["param_criterion"]
    GSSplitters = grid.cv_results_["param_splitter"]
    GSScores = grid.cv_results_["mean_test_score"]
    configIndex = findClosest(GSScores)
    return {"depth":GSMaxDepths[configIndex], "criterion":GSCriterions[configIndex], "splitter":GSSplitters[configIndex], "subSampling":GSSubSamplingRatios[configIndex], "randomState":randomState}
    # bestSettings = []
    # bestResults = []
    # classifier = sklearn.tree.DecisionTreeClassifier(max_depth=1)
    # subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, 0.05, randomState)
    # classifier.fit(subSampledData, subSampledLabels)
    # prediction = classifier.predict(data)
    # preliminary_accuracy = accuracy_score(labels, prediction)
    # if preliminary_accuracy < 0.50:
    #     for max_depth in np.arange(10) + 1:
    #         for subSampling in sorted((np.arange(20, dtype=float) + 1) / 20, reverse=True):
    #             if subSampling > minSubSampling:
    #                 accuracies = np.zeros(50)
    #                 for i in range(50):
    #                     if subSampling != 1.0:
    #                         subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling,
    #                                                                                         randomState)
    #                     else:
    #                         subSampledData, subSampledLabels, = data, labels
    #                     classifier = tree.DecisionTreeClassifier(max_depth=max_depth)
    #                     classifier.fit(subSampledData, subSampledLabels)
    #                     prediction = classifier.predict(data)
    #                     accuracies[i] = accuracy_score(labels, prediction)
    #                 accuracy = np.mean(accuracies)
    #                 if 0.5 < accuracy < 0.60:
    #                     bestSettings.append([max_depth, subSampling])
    #                     bestResults.append(accuracy)
    # else:
    #     preliminary_accuracies = np.zeros(50)
    #     if minSubSampling < 0.01:
    #         for i in range(50):
    #             subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, 0.01, randomState)
    #             classifier.fit(subSampledData, subSampledLabels)
    #             prediction = classifier.predict(data)
    #             preliminary_accuracies[i] = accuracy_score(labels, prediction)
    #     preliminary_accuracy = np.mean(preliminary_accuracies)
    #     if preliminary_accuracy < 0.50:
    #         for subSampling in sorted((np.arange(19, dtype=float) + 1) / 200, reverse=True):
    #             if minSubSampling < subSampling:
    #                 accuracies = np.zeros(50)
    #                 for i in range(50):
    #                     subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling,
    #                                                                                     randomState)
    #                     classifier = tree.DecisionTreeClassifier(max_depth=1)
    #                     classifier.fit(subSampledData, subSampledLabels)
    #                     prediction = classifier.predict(data)
    #                     accuracies[i] = accuracy_score(labels, prediction)
    #                 accuracy = np.mean(accuracies)
    #                 if 0.5 < accuracy < 0.60:
    #                     bestSettings.append([1, subSampling])
    #                     bestResults.append(accuracy)
    #     else:
    #         for subSampling in sorted((np.arange(19, dtype=float) + 1) / 2000, reverse=True):
    #             accuracies = np.zeros(50)
    #             for i in range(50):
    #                 subSampledData, subSampledLabels, subSampledWeights = subSample(data, labels, subSampling,
    #                                                                                 randomState)
    #                 if minSubSampling < subSampling:
    #                     classifier1 = tree.DecisionTreeClassifier(max_depth=1)
    #                     classifier1.fit(subSampledData, subSampledLabels)
    #                     prediction = classifier1.predict(data)
    #                     accuracies[i] = accuracy_score(labels, prediction)
    #             accuracy = np.mean(accuracies)
    #             if 0.5 < accuracy < 0.60:
    #                 bestSettings.append([1, subSampling])
    #                 bestResults.append(accuracy)
    #
    # # assert bestResults != [], "No good settings found for Decision Tree!"
    # if bestResults == []:
    #     bestSetting = None
    # else:
    #     bestSetting = getBestSetting(bestSettings, bestResults)
    # return bestSetting


def getBestSetting(bestSettings, bestResults):
    diffTo52 = 100.0
    bestSettingsIndex = 0
    for resultIndex, result in enumerate(bestResults):
        if abs(0.55 - result) < diffTo52:
            diffTo52 = abs(0.55 - result)
            bestResult = result
            bestSettingsIndex = resultIndex
    logging.debug("\t\tInfo:\t Best Result : " + str(result))

    return map(lambda p: round(p, 4), bestSettings[bestSettingsIndex])
