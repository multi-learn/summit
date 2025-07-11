# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#
import array

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multiclass import _ovr_decision_function
from sklearn.preprocessing import LabelBinarizer

from .dataset import get_samples_views_indices


def get_mc_estim(estimator, random_state, y=None, multiview=False,
                 multiclass=False):
    r"""Used to get a multiclass-compatible estimator if the one in param does not natively support multiclass.
    If perdict_proba is available in the asked estimator, a One Versus Rest wrapper is returned,
    else, a One Versus One wrapper is returned.

    To be able to deal with multiview algorithm, multiview wrappers are implemented separately.

    Parameters
    ----------
    estimator : sklearn-like estimator
        Asked estimator
    y : numpy.array
        The labels of the problem
    random_state : numpy.random.RandomState object
        The random state, used to generate a fake multiclass problem
    multiview : bool
        If True, mutliview-compatible wrappers are returned.

    Returns
    -------
    estimator : sklearn-like estimator
        Either the aksed estimator, or a multiclass-compatible wrapper over the asked estimator
    """
    if (y is not None and np.unique(y).shape[0] > 2) or multiclass:
        if not clone(estimator).accepts_multi_class(random_state):
            if hasattr(estimator, "predict_proba"):
                if multiview:
                    estimator = MultiviewOVRWrapper(estimator)
                else:
                    estimator = OVRWrapper(estimator)
            else:
                if multiview:
                    estimator = MultiviewOVOWrapper(estimator)
                else:
                    estimator = OVOWrapper(estimator)
    return estimator


class MultiClassWrapper:

    # TODO : Has an effect on the init of the sub-classes.
    # @abstractmethod
    # def __init__(self, estimator, **params):
    #     self.estimator = estimator

    def set_params(self, **params):
        r"""
        This function is useful in order for the OV_Wrappers to be transparent
        in terms of parameters.
        If we remove it the parameters have to be specified as estimator__param.
        Witch is not relevant for the platform

        """
        self.estimator.set_params(**params)
        return self

    def get_config(self):
        return "multiclass_adaptation : " + self.__class__.__name__ + \
            ", " + self.estimator.get_config()

    def format_params(self, params, deep=True):
        if hasattr(self, 'estimators_'):
            estim_params = self.estimators_[0].get_params(deep=deep)
            for key, value in params.items():
                if key.startswith("estimator__"):
                    estim_param_key = '__'.join(key.split('__')[1:])
                    params[key] = estim_params[estim_param_key]
            params.pop("estimator")
        return params

    def get_interpretation(self, directory, base_file_name, y_test=None):
        # TODO : Multiclass interpretation
        return "Multiclass wrapper is not interpretable yet"


class MonoviewWrapper(MultiClassWrapper):
    pass


class OVRWrapper(MonoviewWrapper, OneVsRestClassifier):

    def get_params(self, deep=True):
        return self.format_params(
            OneVsRestClassifier.get_params(self, deep=deep), deep=deep)


class OVOWrapper(MonoviewWrapper, OneVsOneClassifier):
    def decision_function(self, X):
        # check_is_fitted(self)

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack([est.predict(Xi)
                                 for est, Xi in zip(self.estimators_, Xs)]).T
        confidences = np.ones(predictions.shape)
        Y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        if self.n_classes_ == 2:
            return Y[:, 1]
        return Y

    def get_params(self, deep=True):
        return self.format_params(
            OneVsOneClassifier.get_params(self, deep=deep), deep=deep)


# The following code is a mutliview adaptation of sklearns multiclass package

def _multiview_fit_binary(estimator, X, y, train_indices,
                          view_indices, classes=None, ):
    # TODO : Verifications des sklearn
    estimator = clone(estimator)
    estimator.fit(X, y, train_indices=train_indices,
                  view_indices=view_indices)
    return estimator


def _multiview_predict_binary(estimator, X, sample_indices, view_indices):
    if is_regressor(estimator):
        return estimator.predict(X, sample_indices=sample_indices,
                                 view_indices=view_indices)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X, sample_indices=sample_indices,
                                        view_indices=view_indices)[:, 1]
    return score


class MultiviewWrapper(MultiClassWrapper):

    def __init__(self, estimator=None, **args):
        super(MultiviewWrapper, self).__init__(estimator=estimator, **args)
        self.short_name = estimator.short_name


class MultiviewOVRWrapper(MultiviewWrapper, OneVsRestClassifier):

    def fit(self, X, y, train_indices=None, view_indices=None):
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = [_multiview_fit_binary(
            self.estimator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]], train_indices=train_indices,
            view_indices=view_indices)
            for i, column in
            enumerate(columns)]
        return self

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        n_samples = len(sample_indices)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _multiview_predict_binary(e, X, sample_indices,
                                                 view_indices)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[argmaxima]
        else:  # pragma: no cover
            if (hasattr(self.estimators_[0], "decision_function") and
                    is_classifier(self.estimators_[0])):
                thresh = 0
            else:
                thresh = .5
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(
                    np.where(_multiview_predict_binary(e, X,
                                                       sample_indices,
                                                       view_indices) > thresh)[
                        0])
                indptr.append(len(indices))

            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)

    def get_params(self, deep=True):
        return self.format_params(
            OneVsRestClassifier.get_params(self, deep=deep), deep=deep)


def _multiview_fit_ovo_binary(estimator, X, y, i, j, train_indices,
                              view_indices):
    cond = np.logical_or(y == i, y == j)
    # y = y[cond]
    y_binary = np.empty(y.shape, np.int_)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(X.get_nb_samples())[cond]
    train_indices = np.intersect1d(train_indices, indcond)
    return _multiview_fit_binary(estimator,
                                 X,
                                 y_binary, train_indices, view_indices,
                                 classes=[i, j]), train_indices


class MultiviewOVOWrapper(MultiviewWrapper, OneVsOneClassifier):

    def get_tags(self):
        """Get  tags of estimateur see  sklearn > 1.6.0 _pairwise attribut  removed """
        if hasattr(self.estimator, "get_tags"):
            return self.estimator.get_tags()
        return {"pairwise": False}

    def fit(self, X, y, train_indices=None, view_indices=None):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        Returns
        -------
        self
        """
        # X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        # check_classification_targets(y)
        train_indices, view_indices = get_samples_views_indices(X,
                                                                train_indices,
                                                                view_indices)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("OneVsOneClassifier can not be fit when only one"
                             " class is present.")
        n_classes = self.classes_.shape[0]
        estimators_indices = list(zip(*([_multiview_fit_ovo_binary(
            self.estimator, X, y, self.classes_[i], self.classes_[j],
            train_indices,
            view_indices
        )
            for i in range(n_classes) for j in range(i + 1, n_classes)
        ])))

        self.estimators_ = estimators_indices[0]
        self.pairwise_indices_ = (
            estimators_indices[1] if self.get_tags()["pairwise"] else None)

        return self

    def predict(self, X, sample_indices=None, view_indices=None):
        """Estimate the best class label for each sample in X.

        This is implemented as ``argmax(decision_function(X), axis=1)`` which
        will return the label of the class with most votes by estimators
        predicting the outcome of a decision for each possible class pair.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        Y = self.multiview_decision_function(X, sample_indices=sample_indices,
                                             view_indices=view_indices)
        if self.n_classes_ == 2:
            return self.classes_[(Y > 0).astype(np.int)]
        return self.classes_[Y.argmax(axis=1)]

    def multiview_decision_function(self, X, sample_indices,
                                    view_indices):  # pragma: no cover
        # check_is_fitted(self)

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            # TODO Gram matrix compatibility
            Xs = [X[:, idx] for idx in indices]
        predictions = np.vstack(
            [est.predict(Xi, sample_indices=sample_indices,
                         view_indices=view_indices)
             for est, Xi in zip(self.estimators_, Xs)]).T
        confidences = np.ones(predictions.shape)
        # confidences = np.vstack([_predict_binary(est, Xi)
        #                          for est, Xi in zip(self.estimators_, Xs)]).T
        Y = _ovr_decision_function(predictions,
                                   confidences, len(self.classes_))
        if self.n_classes_ == 2:
            return Y[:, 1]
        return Y

    def get_params(self, deep=True):
        return self.format_params(
            OneVsOneClassifier.get_params(self, deep=deep), deep=deep)
