import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import metrics
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier, \
    get_accuracy_graph
from ..utils.base import base_boosting_estimators

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "Adaboost"


class Adaboost(AdaBoostClassifier, BaseMonoviewClassifier):
    """
    This class implement a Classifier with adaboost algorithm inherit from sklearn
    AdaBoostClassifier

    Parameters
    ----------

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    n_estimators : int number of estimators

    base_estimator :

    kwargs : others arguments


    Attributes
    ----------
    param_name :

    classed_params :

    distribs :

    weird_strings :

    plotted_metric : selection of metric to plot

    plotted_metric_name : name of the metric to plot

    step_predictions :

    """

    def __init__(self, random_state=None, n_estimators=50,
                 base_estimator=None, base_estimator_config=None, **kwargs):

        base_estimator = BaseMonoviewClassifier.get_base_estimator(self,
                                                                   base_estimator,
                                                  base_estimator_config)
        AdaBoostClassifier.__init__(self,
                                    random_state=random_state,
                                    n_estimators=n_estimators,
                                    base_estimator=base_estimator,
                                    algorithm="SAMME"
                                    )
        self.param_names = ["n_estimators", "base_estimator"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                        base_boosting_estimators]
        self.weird_strings = {"base_estimator": "class_name"}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit adaboost model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        y :  { array-like, shape (n_samples,)
            Target values class labels in classification

        sample_weight :

        Returns
        -------
        self : object
            Returns self.
        """
        begin = time.time()
        AdaBoostClassifier.fit(self, X, y, sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = X.shape
        self.base_predictions = np.array(
            [estim.predict(X) for estim in self.estimators_])
        self.metrics = np.array([self.plotted_metric.score(pred, y) for pred in
                                 self.staged_predict(X)])
        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        Returns
        -------
        predictions : ndarray of shape (n_samples, )
            The estimated labels.
        """
        begin = time.time()
        pred = AdaBoostClassifier.predict(self, X)
        end = time.time()
        self.pred_time = end - begin
        # TODO : mauvaise verif
        if X.shape != self.train_shape:
            self.step_predictions = np.array(
                [step_pred for step_pred in self.staged_predict(X)])
        return pred

    def get_interpretation(self, directory, y_test, multi_class=False):
        interpretString = ""
        interpretString += self.get_feature_importance(directory)
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
