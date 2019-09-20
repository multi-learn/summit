# -*- coding: utf-8 -*-
"""MinCq algorithm.

Related papers:
[1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes (Laviolette et al., 2011)
[2] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (Germain et al., 2015)

"""
from __future__ import print_function, division, absolute_import
import time
from operator import xor

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.validation import check_X_y

from .BoostUtils import ConvexProgram
from ..monoview_utils import change_label_to_zero, change_label_to_minus


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

    def __init__(self, estimators_generator=None, estimators=None, mu=0.001,
                 omega=0.5, use_binary=False, zeta=0, gamma=1, n_neighbors=5):
        if estimators is None:
            estimators = []

        super().__init__(estimators=estimators, voting='soft',
                         flatten_transform=False)
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
        assert xor(bool(self.estimators_generator), bool(
            self.estimators)), "MinCqClassifier: exactly one of estimator_generator or estimators must be used."
        X, y = check_X_y(X, change_label_to_minus(y))

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
            self.clean_me = True

            if isinstance(y, np.ma.MaskedArray):
                transformed_y = np.ma.MaskedArray(self.le_.transform(y), y.mask)
            else:
                # transformed_y = self.le_.transform(y)
                transformed_y = y

            self.estimators_generator.fit(X, transformed_y)
            self.estimators = [('ds{}'.format(i), estimator) for i, estimator in
                               enumerate(self.estimators_generator.estimators_)]
            super().fit(X, y)

        beg = time.time()

        # Preparation and resolution of the quadratic program
        # logger.info("Preparing and solving QP...")
        self.weights = self._solve(X, y)
        if self.clean_me:
            self.estimators = []
        # print(self.weights.shape)
        # print(np.unique(self.weights)[0:10])
        # import pdb;pdb.set_trace()
        self.train_cbound = 1 - (1.0 / X.shape[0]) * (np.sum(
            np.multiply(change_label_to_minus(y),
                        np.average(self._binary_classification_matrix(X),
                                   axis=1, weights=self.weights))) ** 2) / (
                                np.sum(np.average(
                                    self._binary_classification_matrix(X),
                                    axis=1, weights=self.weights) ** 2))
        end = time.time()
        self.train_time = end-beg
        return self

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

    def predict(self, X):
        if not self.estimators:
            self.estimators = [('ds{}'.format(i), estimator) for i, estimator in
                               enumerate(self.estimators_generator.estimators_)]
            self.clean_me = True
        pred = super().predict(X)
        if self.clean_me:
            self.estimators = []
        return change_label_to_zero(pred)

    def _solve(self, X, y):
        y = self.le_.transform(y)

        if self.use_binary:
            assert len(self.le_.classes_) == 2

            # TODO: Review the number of labeled examples when adding back the support for transductive learning.
            classification_matrix = self._binary_classification_matrix(X)

            # We use {-1, 1} labels.
            binary_labels = np.copy(y)
            binary_labels[y == 0] = -1

            multi_matrix = binary_labels.reshape(
                (len(binary_labels), 1)) * classification_matrix

        else:
            multi_matrix = self._multiclass_classification_matrix(X, y)

        n_examples, n_voters = np.shape(multi_matrix)
        ftf = 1.0 / n_examples * multi_matrix.T.dot(multi_matrix)
        yf = np.mean(multi_matrix, axis=0)

        # Objective function.
        objective_matrix = 2 * ftf
        objective_vector = None

        # Equality constraints (first moment of the margin equal to mu, Q sums to one)
        equality_matrix = np.vstack(
            (yf.reshape((1, n_voters)), np.ones((1, n_voters))))
        equality_vector = np.array([self.mu, 1.0])

        # Lower and upper bounds, no quasi-uniformity.
        lower_bound = 0.0
        # TODO: In the case of binary classification, no upper bound will give
        # bad results. Using 1/n works, as it brings back the l_infinity
        # regularization normally given by the quasi-uniformity constraint.
        # upper_bound = 2.0/n_voters
        upper_bound = None

        weights = self._solve_qp(objective_matrix, objective_vector,
                                 equality_matrix, equality_vector, lower_bound,
                                 upper_bound)

        # Keep learning information for further use.
        self.learner_info_ = {}

        # We count the number of non-zero weights, including the implicit voters.
        # TODO: Verify how we define non-zero weights here, could be if the weight is near 1/2n.
        n_nonzero_weights = np.sum(np.asarray(weights) > 1e-12)
        n_nonzero_weights += np.sum(
            np.asarray(weights) < 1.0 / len(self.estimators_) - 1e-12)
        self.learner_info_.update(n_nonzero_weights=n_nonzero_weights)

        return weights

    def _solve_qp(self, objective_matrix, objective_vector, equality_matrix,
                  equality_vector, lower_bound, upper_bound):
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
        import time
        beg = time.time()
        # We first fit and learn the weights.
        super().fit(X, y)

        # Validations
        if isinstance(y, np.ma.MaskedArray):
            assert len(self.classes_[np.where(np.logical_not(
                self.classes_.mask))]) == 2, "RegularizedBinaryMinCqClassifier: only supports binary classification."
        else:
            assert len(
                self.classes_), "RegularizedBinaryMinCqClassifier: only supports binary classification."

        # Then we "reverse" the negative weights and their associated voter's output.
        for i, weight in enumerate(self.weights):
            if weight < 0:
                # logger.debug("Reversing decision of a binary voter")
                self.weights[i] *= -1
                self.estimators_[i].reverse_decision()
        end = time.time()
        self.train_time = end - beg
        return self

    def _solve(self, X, y):
        if isinstance(y, np.ma.MaskedArray):
            y = np.ma.MaskedArray(self.le_.transform(y), y.mask)
        else:
            y = self.le_.transform(y)

        classification_matrix = self._binary_classification_matrix(X)
        n_examples, n_voters = np.shape(classification_matrix)

        if self.zeta == 0:
            np.transpose(classification_matrix)
            ftf = np.dot(np.transpose(classification_matrix),
                         classification_matrix)
        else:
            I = np.eye(n_examples)
            L = build_laplacian(X, n_neighbors=self.n_neighbors)
            ftf = classification_matrix.T.dot(
                I + (self.zeta / n_examples) * L).dot(classification_matrix)

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

        try:
            weights = self._solve_qp(objective_matrix, objective_vector,
                                     equality_matrix, equality_vector,
                                     lower_bound, upper_bound)
        except ValueError as e:
            if "domain error" in e.args:
                weights = np.ones(len(self.estimators_))

        # Keep learning information for further use.
        self.learner_info_ = {}

        # We count the number of non-zero weights, including the implicit voters.
        # TODO: Verify how we define non-zero weights here, could be if the weight is near 1/2n.
        n_nonzero_weights = np.sum(np.asarray(weights) > 1e-12)
        n_nonzero_weights += np.sum(
            np.asarray(weights) < 1.0 / len(self.estimators_) - 1e-12)
        self.learner_info_.update(n_nonzero_weights=n_nonzero_weights)

        # Conversion of the weights of the n first voters to weights on the implicit 2n voters.
        # See Section 7.1 of [2] for an explanation.
        # return np.array([2 * q - 1.0 / len(self.estimators_) for q in weights])
        return np.array(weights)


def build_laplacian(X, n_neighbors=None):
    clf = SpectralEmbedding(n_neighbors=n_neighbors)
    clf.fit(X)
    w = clf.affinity_matrix_
    laplacian = graph_laplacian(w, normed=True)
    return laplacian
