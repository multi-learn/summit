import scipy
import logging
import numpy.ma as ma
from collections import defaultdict
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import time

from ..Monoview.MonoviewUtils import CustomRandint, CustomUniform, BaseMonoviewClassifier
from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.CQBoostUtils import ColumnGenerationClassifier

class ColumnGenerationClassifierv2(ColumnGenerationClassifier):

    def __init__(self, mu=0.01, epsilon=1e-06, random_state=None):
        super(ColumnGenerationClassifierv2, self).__init__(mu=mu, epsilon=epsilon, random_state=random_state)

    def initialize(self):
        self.weights_ = []
        self.edge_scores = []
        self.alphas = []

    def update_values(self, h_values=None, worst_h_index=None, alpha=None, w=None):
        self.edge_scores.append(h_values[worst_h_index])
        self.alphas.append(alpha)
        self.weights_.append(w[-1])

    def get_margins(self, w=None):
        self.weights = np.array(self.weights_)
        self.final_vote_weights = np.array([np.prod(1 - self.weights[t + 1:]) * self.weights_[t] if t <
                                                                                                    self.weights.shape[
                                                                                                        0] - 1 else
                                            self.weights[t] for t in range(self.weights.shape[0])])
        margins = np.squeeze(np.asarray(np.matmul(self.classification_matrix[:, self.chosen_columns_],
                                                  self.final_vote_weights)))
        return margins

    def compute_weights_(self, w=None):
        self.weights_ = np.array(self.weights_)
        self.final_vote_weights = np.array([np.prod(1 - self.weights_[t + 1:]) * self.weights_[t] if t <
                                                                                                     self.weights_.shape[
                                                                                                         0] - 1 else
                                            self.weights_[t] for t in range(self.weights_.shape[0])])
        self.weights_ = self.final_vote_weights

    def get_matrix_to_optimize(self, y_kernel_matrix, w=None):
        m = self.n_total_examples
        if w is not None:
            matrix_to_optimize = np.concatenate((np.matmul(self.matrix_to_optimize, w).reshape((m, 1)),
                                                                  y_kernel_matrix[:, self.chosen_columns_[-1]].reshape((m, 1))),
                                                                 axis=1)
        else:
            matrix_to_optimize = y_kernel_matrix[:, self.chosen_columns_[-1]].reshape((m, 1))
        return matrix_to_optimize



class CQBoostv2(ColumnGenerationClassifierv2, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, epsilon=1e-06, **kwargs):
        super(CQBoostv2, self).__init__(
            random_state=random_state,
            mu=mu,
            epsilon=epsilon
        )
        self.param_names = ["mu", "epsilon"]
        self.distribs = [CustomUniform(loc=0.5, state=1.0, multiplier="e-"),
                         CustomRandint(low=1, high=15, multiplier="e-")]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        return getInterpretBase(self, directory, "CQBoostv2", self.weights_,)

    def get_name_for_fusion(self):
        return "CQB2"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"mu": args.CQB_mu,
                  "epsilon": args.CQB_epsilon}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": 10**-randomState.uniform(0.5, 1.5),
                          "epsilon": 10**-randomState.randint(1, 15)})
    return paramsSet


# class CQBoostv2(CqBoostClassifierv2):
#
#     def __init__(self, random_state, **kwargs):
#         super(CQBoostv2, self).__init__(
#             mu=kwargs['mu'],
#             epsilon=kwargs['epsilon'],
#             n_max_iterations= kwargs['n_max_iterations'],
#             )
#
#     def canProbas(self):
#         """Used to know if the classifier can return label probabilities"""
#         return False
#
#     def paramsToSrt(self, nIter=1):
#         """Used for weighted linear early fusion to generate random search sets"""
#         paramsSet = []
#         for _ in range(nIter):
#             paramsSet.append({"mu": 0.001,
#                               "epsilon": 1e-08,
#                               "n_max_iterations": None})
#         return paramsSet
#
#     def getKWARGS(self, args):
#         """Used to format kwargs for the parsed args"""
#         kwargsDict = {}
#         kwargsDict['mu'] = 0.001
#         kwargsDict['epsilon'] = 1e-08
#         kwargsDict['n_max_iterations'] = None
#         return kwargsDict
#
#     def genPipeline(self):
#         return Pipeline([('classifier', CqBoostClassifierv2())])
#
#     def genParamsDict(self, randomState):
#         return {"classifier__mu": [0.001],
#                 "classifier__epsilon": [1e-08],
#                 "classifier__n_max_iterations": [None]}
#
#     def genBestParams(self, detector):
#         return {"mu": detector.best_params_["classifier__mu"],
#                 "epsilon": detector.best_params_["classifier__epsilon"],
#                 "n_max_iterations": detector.best_params_["classifier__n_max_iterations"]}
#
#     def genParamsFromDetector(self, detector):
#         nIter = len(detector.cv_results_['param_classifier__mu'])
#         return [("mu", np.array([0.001 for _ in range(nIter)])),
#                 ("epsilon", np.array(detector.cv_results_['param_classifier__epsilon'])),
#                 ("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]
#
#     def getConfig(self, config):
#         if type(config) is not dict:  # Used in late fusion when config is a classifier
#             return "\n\t\t- CQBoost with mu : " + str(config.mu) + ", epsilon : " + str(
#                 config.epsilon + ", n_max_iterations : " + str(config.n_max_iterations))
#         else:
#             return "\n\t\t- CQBoost with mu : " + str(config["mu"]) + ", epsilon : " + str(
#                    config["epsilon"] + ", n_max_iterations : " + str(config["n_max_iterations"]))
#
#
#     def getInterpret(self, classifier, directory):
#         interpretString = ""
#         return interpretString
#
#
# def canProbas():
#     return False
#
#
# def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
#     """Used to fit the monoview classifier with the args stored in kwargs"""
#     start = time.time()
#     classifier = CqBoostClassifierv2(mu=kwargs['mu'],
#                                    epsilon=kwargs['epsilon'],
#                                    n_max_iterations=kwargs["n_max_iterations"],)
#                                    # random_state=randomState)
#     classifier.fit(DATASET, CLASS_LABELS)
#     end = time.time()
#     classifier.train_time =end-start
#     return classifier
#
#
# def paramsToSet(nIter, randomState):
#     """Used for weighted linear early fusion to generate random search sets"""
#     paramsSet = []
#     for _ in range(nIter):
#         paramsSet.append({"mu": randomState.uniform(1e-02, 10**(-0.5)),
#                           "epsilon": 10**-randomState.randint(1, 15),
#                           "n_max_iterations": None})
#     return paramsSet
#
#
# def getKWARGS(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {}
#     kwargsDict['mu'] = args.CQB2_mu
#     kwargsDict['epsilon'] = args.CQB2_epsilon
#     kwargsDict['n_max_iterations'] = None
#     return kwargsDict
#
#
# def genPipeline():
#     return Pipeline([('classifier', CqBoostClassifierv2())])
#
#
# def genParamsDict(randomState):
#     return {"classifier__mu": CustomUniform(loc=.5, state=2, multiplier='e-'),
#                 "classifier__epsilon": CustomRandint(low=1, high=15, multiplier='e-'),
#                 "classifier__n_max_iterations": [None]}
#
#
# def genBestParams(detector):
#     return {"mu": detector.best_params_["classifier__mu"],
#                 "epsilon": detector.best_params_["classifier__epsilon"],
#                 "n_max_iterations": detector.best_params_["classifier__n_max_iterations"]}
#
#
# def genParamsFromDetector(detector):
#     nIter = len(detector.cv_results_['param_classifier__mu'])
#     return [("mu", np.array([0.001 for _ in range(nIter)])),
#             ("epsilon", np.array(detector.cv_results_['param_classifier__epsilon'])),
#             ("n_max_iterations", np.array(detector.cv_results_['param_classifier__n_max_iterations']))]
#
#
# def getConfig(config):
#     if type(config) is not dict:  # Used in late fusion when config is a classifier
#         return "\n\t\t- CQBoostv2 with mu : " + str(config.mu) + ", epsilon : " + str(
#             config.epsilon) + ", n_max_iterations : " + str(config.n_max_iterations)
#     else:
#         return "\n\t\t- CQBoostv2 with mu : " + str(config["mu"]) + ", epsilon : " + str(
#             config["epsilon"]) + ", n_max_iterations : " + str(config["n_max_iterations"])
#
#
# def getInterpret(classifier, directory):
#     return getInterpretBase(classifier, directory, "CQBoostv2", classifier.final_vote_weights)

