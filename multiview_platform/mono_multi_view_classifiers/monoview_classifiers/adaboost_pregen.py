import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import metrics
from ..monoview.additions.BoostUtils import get_accuracy_graph
from ..monoview.additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier, \
    change_label_to_zero

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "AdaboostPregen"

class AdaboostPregen(AdaBoostClassifier, BaseMonoviewClassifier,
                     PregenClassifier):
    """

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    n_estimators : int number of estimators (default : 50)

    base_estimator :

    n_stumps : int (default : 1)

    estimators_generator : str, (default : "Stumps")

    max_depth : int (default : 1)

    self_complemeted : bool, (default : True)

    kwargs : others arguments


    Attributes
    ----------

    param_names : list of parameters names

    classed_params :  list of parameters names

    distribs :

    weird_strings :

    plotted_metric

    plotted_metric_name : str name of plotted  metric

    step_predictions :

    estimators_generator :

    max_depth :

    n_stumps :

    self_complemented :

    """
    def __init__(self, random_state=None, n_estimators=50,
                 base_estimator=None, n_stumps=1, estimators_generator="Stumps",
                 max_depth=1, self_complemeted=True,
                 **kwargs):
        super(AdaboostPregen, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            algorithm="SAMME"
        )
        self.param_names = ["n_estimators", "base_estimator", "n_stumps",
                            "estimators_generator", "max_depth",
                            "random_state"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500),
                         [DecisionTreeClassifier(max_depth=1)], [n_stumps],
                         ["Stumps", "Tree"], CustomRandint(low=1, high=5),
                         [random_state]]
        self.weird_strings = {"base_estimator": "class_name"}
        self.plotted_metric = metrics.zero_one_loss
        self.plotted_metric_name = "zero_one_loss"
        self.step_predictions = None
        self.estimators_generator = estimators_generator
        self.max_depth = max_depth
        self.n_stumps = n_stumps
        self.self_complemented = self_complemeted

    def fit(self, X, y, sample_weight=None):
        """
        Fit the AdaboostPregen

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        y :  { array-like, shape (n_samples,)
            Target values class labels in classification

        sample_weight :


        """
        begin = time.time()
        pregen_X, pregen_y = self.pregen_voters(X, y)
        super(AdaboostPregen, self).fit(pregen_X, pregen_y,
                                        sample_weight=sample_weight)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = pregen_X.shape
        self.base_predictions = np.array(
            [change_label_to_zero(estim.predict(pregen_X)) for estim in
             self.estimators_])
        self.metrics = np.array(
            [self.plotted_metric.score(change_label_to_zero(pred), y) for pred
             in self.staged_predict(pregen_X)])

        self.bounds = np.array([np.prod(
            np.sqrt(1 - 4 * np.square(0.5 - self.estimator_errors_[:i + 1])))
                                for i in
                                range(self.estimator_errors_.shape[0])])

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     True
    #     """
    #     return True

    def predict(self, X):
        """

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------

        """
        begin = time.time()
        pregen_X, _ = self.pregen_voters(X)
        pred = super(AdaboostPregen, self).predict(pregen_X)
        end = time.time()
        self.pred_time = end - begin
        if pregen_X.shape != self.train_shape:
            self.step_predictions = np.array(
                [change_label_to_zero(step_pred) for step_pred in
                 self.staged_predict(pregen_X)])
        return change_label_to_zero(pred)

    # def set_params(self, **params):
    #     super().set_params(params)
    #     self.random_state = params["random_state"]
    #     self.n_stumps_per_attribute = params["n_tumps"]
    #     return self

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
        get_accuracy_graph(step_test_metrics, "AdaboostPregen",
                           directory + "test_metrics.png",
                           self.plotted_metric_name, set="test")
        # get_accuracy_graph(self.metrics, "AdaboostPregen",
        #                    directory + "metrics.png", self.plotted_metric_name,
        #                    bounds=list(self.bounds),
        #                    bound_name="boosting bound")
        np.savetxt(directory + "test_metrics.csv", step_test_metrics,
                   delimiter=',')
        np.savetxt(directory + "train_metrics.csv", self.metrics, delimiter=',')
        np.savetxt(directory + "times.csv",
                   np.array([self.train_time, self.pred_time]), delimiter=',')
        np.savetxt(directory + "times_iter.csv",
                   np.array([self.train_time, len(self.estimator_weights_)]), delimiter=',')
        return interpretString

# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {'n_estimators': args.AdP_n_est,
#                   'base_estimator': [DecisionTreeClassifier(max_depth=1)],
#                   'n_stumps': args.AdP_stumps}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": random_state.randint(1, 500),
                          "base_estimator": None})
    return paramsSet
