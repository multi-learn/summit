import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import metrics
from ..monoview.additions.BoostUtils import get_accuracy_graph
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "Adaboost"


class Adaboost(AdaBoostClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_estimators=50,
                 base_estimator=None, **kwargs):
        super(Adaboost, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            algorithm="SAMME"
        )
        self.param_names = ["n_estimators", "base_estimator"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                         [DecisionTreeClassifier(max_depth=1)]]
        self.weird_strings = {"base_estimator": "class_name"}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None

    def fit(self, X, y, sample_weight=None):
        begin = time.time()
        super(Adaboost, self).fit(X, y, sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = X.shape
        self.base_predictions = np.array(
            [estim.predict(X) for estim in self.estimators_])
        self.metrics = np.array([self.plotted_metric.score(pred, y) for pred in
                                 self.staged_predict(X)])

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def predict(self, X):
        begin = time.time()
        pred = super(Adaboost, self).predict(X)
        end = time.time()
        self.pred_time = end - begin
        if X.shape != self.train_shape:
            self.step_predictions = np.array(
                [step_pred for step_pred in self.staged_predict(X)])
        return pred

    def getInterpret(self, directory, y_test):
        interpretString = ""
        interpretString += self.getFeatureImportance(directory)
        interpretString += "\n\n Estimator error | Estimator weight\n"
        interpretString += "\n".join(
            [str(error) + " | " + str(weight / sum(self.estimator_weights_)) for
             error, weight in
             zip(self.estimator_errors_, self.estimator_weights_)])
        step_test_metrics = np.array(
            [self.plotted_metric.score(y_test, step_pred) for step_pred in
             self.step_predictions])
        get_accuracy_graph(step_test_metrics, "Adaboost",
                           directory + "test_metrics.png",
                           self.plotted_metric_name, set="test")
        np.savetxt(directory + "test_metrics.csv", step_test_metrics,
                   delimiter=',')
        np.savetxt(directory + "train_metrics.csv", self.metrics, delimiter=',')
        np.savetxt(directory + "times.csv",
                   np.array([self.train_time, self.pred_time]), delimiter=',')
        return interpretString

#
# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {'n_estimators': args.Ada_n_est,
#                   'base_estimator': [DecisionTreeClassifier(max_depth=1)]}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": random_state.randint(1, 500),
                          "base_estimator": None})
    return paramsSet
