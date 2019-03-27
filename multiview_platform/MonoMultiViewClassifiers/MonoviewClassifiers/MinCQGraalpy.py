# -*- coding: utf-8 -*-
"""MinCq algorithm.

Related papers:
[1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes (Laviolette et al., 2011)
[2] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (Germain et al., 2015)

"""
from __future__ import print_function, division, absolute_import
import logging
from operator import xor

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y
from sklearn.ensemble import VotingClassifier
from sklearn.manifold import SpectralEmbedding
from sklearn.utils.graph import graph_laplacian
from sklearn.preprocessing import LabelEncoder


from ..Monoview.Additions.BoostUtils import ConvexProgram, StumpsClassifiersGenerator
from ..Monoview.MonoviewUtils import BaseMonoviewClassifier, CustomUniform
from ..Metrics import zero_one_loss

# logger = logging.getLogger('MinCq')

class MinCqClassifier(VotingClassifier):
    """
    Base MinCq algorithm learner. See [1, 2].
    This version is an attempt of creating a more general version of MinCq, that handles multiclass classfication.
    For binary classification, use RegularizedMinCqClassifer.

    Parameters
    ----------
    mu : float
        The fixed value of the first moment of the margin.

    """
    def __init__(self, estimators_generator=None, estimators=None, mu=0.001, omega=0.5, use_binary=False, zeta=0, gamma=1, n_neighbors=5):
        if estimators is None:
            estimators = []

        super().__init__(estimators=estimators, voting='soft')
        self.estimators_generator = estimators_generator
        self.mu = mu
        self.omega = omega
        self.use_binary = use_binary
        self.zeta = zeta
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the estimators and learn the weights.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values. If y is a masked-array (numpy.ma), the masked values are unlabeled examples.

        Returns
        -------
        self : object

        """
        # Validations
        assert 0 < self.mu <= 1, "MinCqClassifier: mu parameter must be in (0, 1]"
        assert xor(bool(self.estimators_generator), bool(self.estimators)), "MinCqClassifier: exactly one of estimator_generator or estimators must be used."
        X, y = check_X_y(X, y)

        # Fit the estimators using VotingClassifier's fit method. This will also fit a LabelEncoder that can be
        # used to "normalize" labels (0, 1, 2, ...). In the case of binary classification, the two classes will be 0 and 1.
        # First, ensure that the weights are reset to None (as cloning a VotingClassifier keeps the weights)
        self.weights = None
        # TODO: Ensure estimators can deal with masked arrays

        # If we use an estimator generator, use the data-dependant estimator generator to generate them, and fit again.
        if self.estimators:
            super().fit(X, y)

        else:
            self.le_ = LabelEncoder()
            self.le_.fit(y)

            if isinstance(y, np.ma.MaskedArray):
                transformed_y = np.ma.MaskedArray(self.le_.transform(y), y.mask)
            else:
                transformed_y = self.le_.transform(y)

            self.estimators_generator.fit(X, transformed_y)
            self.estimators = [('ds{}'.format(i), estimator) for i, estimator in enumerate(self.estimators_generator.estimators_)]
            super().fit(X, y)

            # We clean the estimators attribute (as we do not want it to be cloned later)
            # self.estimators_ = []

        # logger.info("Training started...")
        # logger.info("Training dataset shape: {}".format(str(np.shape(X))))
        # logger.info("Number of voters: {}".format(len(self.estimators_)))

        # Preparation and resolution of the quadratic program
        # logger.info("Preparing and solving QP...")
        self.weights = self._solve(X, y)

        return self

    # def evaluate_metrics(self, X, y, metrics_list=None, functions_list=None):
    #     if metrics_list is None:
    #         metrics_list = [zero_one_loss]
    #
    #     if functions_list is None:
    #         functions_list = []
    #     else:
    #         raise NotImplementedError
    #
    #     # Predict, evaluate metrics.
    #     predictions = self.predict(X)
    #     metrics_results = {metric.__name__: metric(y, predictions) for metric in metrics_list}
    #
    #     metrics_dataframe = ResultsDataFrame([metrics_results])
    #     return metrics_dataframe

    def _binary_classification_matrix(self, X):
        probas = self.transform(X)
        predicted_labels = np.argmax(probas, axis=2)
        predicted_labels[predicted_labels == 0] = -1
        values = np.max(probas, axis=2)
        return (predicted_labels * values).T

    def _multiclass_classification_matrix(self, X, y):
        probas = self.transform(X).swapaxes(0, 1)
        matrix = probas[np.arange(probas.shape[0]), :, y]

        return (matrix - self.omega)

    def _solve(self, X, y):
        y = self.le_.transform(y)

        if self.use_binary:
            assert len(self.le_.classes_) == 2

            # TODO: Review the number of labeled examples when adding back the support for transductive learning.
            classification_matrix = self._binary_classification_matrix(X)

            # We use {-1, 1} labels.
            binary_labels = np.copy(y)
            binary_labels[y == 0] = -1

            multi_matrix = binary_labels.reshape((len(binary_labels), 1)) * classification_matrix

        else:
            multi_matrix = self._multiclass_classification_matrix(X, y)

        n_examples, n_voters = np.shape(multi_matrix)
        ftf = 1.0 / n_examples * multi_matrix.T.dot(multi_matrix)
        yf = np.mean(multi_matrix, axis=0)

        # Objective function.
        objective_matrix = 2 * ftf
        objective_vector = None

        # Equality constraints (first moment of the margin equal to mu, Q sums to one)
        equality_matrix = np.vstack((yf.reshape((1, n_voters)), np.ones((1, n_voters))))
        equality_vector = np.array([self.mu, 1.0])

        # Lower and upper bounds, no quasi-uniformity.
        lower_bound = 0.0
        # TODO: In the case of binary classification, no upper bound will give
        # bad results. Using 1/n works, as it brings back the l_infinity
        # regularization normally given by the quasi-uniformity constraint.
        # upper_bound = 2.0/n_voters
        upper_bound = None

        weights = self._solve_qp(objective_matrix, objective_vector, equality_matrix, equality_vector, lower_bound, upper_bound)

        # Keep learning information for further use.
        self.learner_info_ = {}

        # We count the number of non-zero weights, including the implicit voters.
        # TODO: Verify how we define non-zero weights here, could be if the weight is near 1/2n.
        n_nonzero_weights = np.sum(np.asarray(weights) > 1e-12)
        n_nonzero_weights += np.sum(np.asarray(weights) < 1.0 / len(self.estimators_) - 1e-12)
        self.learner_info_.update(n_nonzero_weights=n_nonzero_weights)

        return weights

    def _solve_qp(self, objective_matrix, objective_vector, equality_matrix, equality_vector, lower_bound, upper_bound):
        try:
            qp = ConvexProgram()
            qp.quadratic_func, qp.linear_func = objective_matrix, objective_vector
            qp.add_equality_constraints(equality_matrix, equality_vector)
            qp.add_lower_bound(lower_bound)
            qp.add_upper_bound(upper_bound)
            return qp.solve()

        except Exception:
            # logger.warning("Error while solving the quadratic program.")
            raise


class RegularizedBinaryMinCqClassifier(MinCqClassifier):
    """MinCq, version published in [1] and [2], where the regularization comes from the enforced quasi-uniformity
    of the posterior distributino on the symmetric hypothesis space. This version only works with {-1, 1} labels.

    [1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes (Laviolette et al., 2011)
    [2] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (Germain et al., 2015)

    """
    def fit(self, X, y):
        # We first fit and learn the weights.
        super().fit(X, y)

        # Validations
        if isinstance(y, np.ma.MaskedArray):
            assert len(self.classes_[np.where(np.logical_not(self.classes_.mask))]) == 2, "RegularizedBinaryMinCqClassifier: only supports binary classification."
        else:
            assert len(self.classes_), "RegularizedBinaryMinCqClassifier: only supports binary classification."

        # Then we "reverse" the negative weights and their associated voter's output.
        for i, weight in enumerate(self.weights):
            if weight < 0:
                # logger.debug("Reversing decision of a binary voter")
                self.weights[i] *= -1
                self.estimators_[i].reverse_decision()

        return self

    def _solve(self, X, y):
        if isinstance(y, np.ma.MaskedArray):
            y = np.ma.MaskedArray(self.le_.transform(y), y.mask)
        else:
            y = self.le_.transform(y)

        classification_matrix = self._binary_classification_matrix(X)
        n_examples, n_voters = np.shape(classification_matrix)

        if self.zeta == 0:
            ftf = classification_matrix.T.dot(classification_matrix)
        else:
            I = np.eye(n_examples)
            L = build_laplacian(X, n_neighbors=self.n_neighbors)
            ftf = classification_matrix.T.dot(I + (self.zeta / n_examples) * L).dot(classification_matrix)

        # We use {-1, 1} labels.
        binary_labels = np.ma.copy(y)
        binary_labels[np.ma.where(y == 0)] = -1

        # Objective function.
        ftf_mean = np.mean(ftf, axis=1)
        objective_matrix = 2.0 / n_examples * ftf
        objective_vector = -1.0 / n_examples * ftf_mean.T

        # Equality constraint: first moment of the margin fixed to mu, only using labeled examples.
        if isinstance(y, np.ma.MaskedArray):
            labeled = np.where(np.logical_not(y.mask))[0]
            binary_labels = binary_labels[labeled]
        else:
            labeled = range(len(y))

        yf = binary_labels.T.dot(classification_matrix[labeled])
        yf_mean = np.mean(yf)
        equality_matrix = 2.0 / len(labeled) * yf
        equality_vector = self.mu + 1.0 / len(labeled) * yf_mean

        # Lower and upper bounds (quasi-uniformity constraints)
        lower_bound = 0.0
        upper_bound = 1.0 / n_voters

        weights = self._solve_qp(objective_matrix, objective_vector, equality_matrix, equality_vector, lower_bound, upper_bound)

        # Keep learning information for further use.
        self.learner_info_ = {}

        # We count the number of non-zero weights, including the implicit voters.
        # TODO: Verify how we define non-zero weights here, could be if the weight is near 1/2n.
        n_nonzero_weights = np.sum(np.asarray(weights) > 1e-12)
        n_nonzero_weights += np.sum(np.asarray(weights) < 1.0 / len(self.estimators_) - 1e-12)
        self.learner_info_.update(n_nonzero_weights=n_nonzero_weights)

        # Conversion of the weights of the n first voters to weights on the implicit 2n voters.
        # See Section 7.1 of [2] for an explanation.
        return np.array([2 * q - 1.0 / len(self.estimators_) for q in weights])

    # def evaluate_metrics(self, X, y, metrics_list=None, functions_list=None):
    #     if metrics_list is None:
    #         metrics_list = [zero_one_loss]
    #
    #     if functions_list is None:
    #         functions_list = []
    #
    #     # Transductive setting: we only predict the X for labeled y
    #     if isinstance(y, np.ma.MaskedArray):
    #         labeled = np.where(np.logical_not(y.mask))[0]
    #         X = np.array(X[labeled])
    #         y = np.array(y[labeled])
    #
    #     # Predict, evaluate metrics.
    #     predictions = self.predict(X)
    #     metrics_results = {metric.__name__: metric(y, predictions) for metric in metrics_list}
    #
    #     # TODO: Repair in the case of non-{-1, 1} labels.
    #     assert set(y) == {-1, 1}
    #     classification_matrix = self._binary_classification_matrix(X)
    #
    #     for function in functions_list:
    #         metrics_results[function.__name__] = function(classification_matrix, y, self.weights)
    #
    #     metrics_dataframe = ResultsDataFrame([metrics_results])
    #     return metrics_dataframe


def build_laplacian(X, n_neighbors=None):
    clf = SpectralEmbedding(n_neighbors=n_neighbors)
    clf.fit(X)
    w = clf.affinity_matrix_
    laplacian = graph_laplacian(w, normed=True)
    return laplacian


class MinCQGraalpy(RegularizedBinaryMinCqClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, self_complemented=True, n_stumps_per_attribute=10 , **kwargs):
        super(MinCQGraalpy, self).__init__(mu=mu,
            estimators_generator=StumpsClassifiersGenerator(n_stumps_per_attribute=n_stumps_per_attribute, self_complemented=self_complemented),
        )
        self.param_names = ["mu"]
        self.distribs = [CustomUniform(loc=0.5, state=2.0, multiplier="e-"),
                         ]
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.classed_params = []
        self.weird_strings = {}
        self.random_state = random_state
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def set_params(self, **params):
        self.mu = params["mu"]

    def get_params(self, deep=True):
        return {"random_state":self.random_state, "mu":self.mu}

    def getInterpret(self, directory, y_test):
        interpret_string = ""
        # interpret_string += "Train C_bound value : "+str(self.cbound_train)
        # y_rework = np.copy(y_test)
        # y_rework[np.where(y_rework==0)] = -1
        # interpret_string += "\n Test c_bound value : "+str(self.majority_vote.cbound_value(self.x_test, y_rework))
        return interpret_string

    def get_name_for_fusion(self):
        return "MCG"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"mu":args.MCG_mu,
                  "n_stumps_per_attribute":args.MCG_stumps}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet


# if __name__ == '__main__':
#     # Example usage.
#     from sklearn.datasets import load_iris
#     from sklearn.cross_validation import train_test_split
#     from graalpy.utils.majority_vote import StumpsClassifiersGenerator
#
#     # Load data, change {0, 1, 2} labels to {-1, 1}
#     iris = load_iris()
#     iris.target[np.where(iris.target == 0)] = -1
#     iris.target[np.where(iris.target == 2)] = 1
#     x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
#
#     # Fit MinCq
#     clf = RegularizedBinaryMinCqClassifier(estimators_generator=StumpsClassifiersGenerator())
#     clf.fit(x_train, y_train)
#
#     # Compare the best score of individual classifiers versus the score of the learned majority vote.
#     print("Best training risk of individual voters: {:.4f}".format(1 - max([e.score(x_train, y_train) for e in clf.estimators_])))
#     print("Training risk of the majority vote outputted by MinCq: {:.4f}".format(1 - clf.score(x_train, y_train)))
#     print()
#     print("Best testing risk of individual voters: {:.4f}".format(1 - max([e.score(x_test, y_test) for e in clf.estimators_])))
#     print("Testing risk of the majority vote outputted by MinCq: {:.4f}".format(1 - clf.score(x_test, y_test)))
