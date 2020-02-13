import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import metrics
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier, get_accuracy_graph

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "GradientBoosting"

class CustomDecisionTree(DecisionTreeClassifier):
    def predict(self, X, check_input=True):
        y_pred = super(CustomDecisionTree, self).predict(X,
                                                         check_input=check_input)
        return y_pred.reshape((y_pred.shape[0], 1)).astype(float)


class GradientBoosting(GradientBoostingClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, loss="exponential", max_depth=1.0,
                 n_estimators=100,
                 init=CustomDecisionTree(max_depth=1),
                 **kwargs):
        super(GradientBoosting, self).__init__(
            loss=loss,
            max_depth=max_depth,
            n_estimators=n_estimators,
            init=init,
            random_state=random_state
        )
        self.param_names = ["n_estimators", ]
        self.classed_params = []
        self.distribs = [CustomRandint(low=50, high=500), ]
        self.weird_strings = {}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None

    def fit(self, X, y, sample_weight=None, monitor=None):
        begin = time.time()
        super(GradientBoosting, self).fit(X, y, sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = X.shape
        self.base_predictions = np.array(
            [estim[0].predict(X) for estim in self.estimators_])
        self.metrics = np.array(
            [self.plotted_metric.score(pred, y) for pred in
             self.staged_predict(X)])
        # self.bounds = np.array([np.prod(
        #     np.sqrt(1 - 4 * np.square(0.5 - self.estimator_errors_[:i + 1]))) for i
        #                         in range(self.estimator_errors_.shape[0])])
        return self

    def predict(self, X):
        begin = time.time()
        pred = super(GradientBoosting, self).predict(X)
        end = time.time()
        self.pred_time = end - begin
        if X.shape != self.train_shape:
            self.step_predictions = np.array(
                [step_pred for step_pred in self.staged_predict(X)])
        return pred

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return False

    def get_interpretation(self, directory, y_test, multi_class=False):
        interpretString = ""
        if multi_class:
            return interpretString
        else:
            interpretString += self.get_feature_importance(directory)
            step_test_metrics = np.array(
                [self.plotted_metric.score(y_test, step_pred) for step_pred in
                 self.step_predictions])
            get_accuracy_graph(step_test_metrics, "AdaboostClassic",
                               directory + "test_metrics.png",
                               self.plotted_metric_name, set="test")
            get_accuracy_graph(self.metrics, "AdaboostClassic",
                               directory + "metrics.png", self.plotted_metric_name)
            np.savetxt(directory + "test_metrics.csv", step_test_metrics,
                       delimiter=',')
            np.savetxt(directory + "train_metrics.csv", self.metrics, delimiter=',')
            np.savetxt(directory + "times.csv",
                       np.array([self.train_time, self.pred_time]), delimiter=',')
            return interpretString


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"n_estimators": args.GB_n_est, }
#     return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": randomState.randint(50, 500), })
    return paramsSet
