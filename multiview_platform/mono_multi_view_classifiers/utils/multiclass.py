import array

import numpy as np
import scipy.sparse as sp
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multiclass import _ovr_decision_function
from sklearn.preprocessing import LabelBinarizer

from .dataset import get_examples_views_indices


# def gen_multiclass_labels(labels, multiclass_method, splits):
#     r"""Used to gen the train/test splits and to set up the framework of the adaptation of a multiclass dataset
#     to biclass algorithms.
#
#     First, the function checks whether the dataset is really multiclass.
#
#     Then, it generates all the possible couples of different labels in order to perform one versus one classification.
#
#     For each combination, it selects the examples in the training sets (for each statistical iteration) that have their
#     label in the combination and does the same for the testing set. It also saves the multiclass testing set in order to
#     use multiclass metrics on the decisions.
#
#     Lastly, it creates a new array of biclass labels (0/1) for the biclass classifications used in oneVersusOne
#
#     Parameters
#     ----------
#     labels : numpy.ndarray
#         Name of the database.
#     multiclass_method : string
#         The name of the multiclass method used (oneVersusOne, oneVersusAll, ...).
#     splits : list of lists of numpy.ndarray
#         For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
#         the ones of the testing set.
#
#     Returns
#     -------
#     multiclass_labels : list of lists of numpy.ndarray
#         For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
#         indices for the biclass training set, the ones for the biclass testing set and the ones for the
#         multiclass testing set.
#
#     labels_indices : list of lists of numpy.ndarray
#         Each original couple of different labels.
#
#     indices_multiclass : list of lists of numpy.ndarray
#         For each combination, contains a biclass labels numpy.ndarray with the 0/1 labels of combination.
#     """
#     if multiclass_method == "oneVersusOne":
#         nb_labels = len(set(list(labels)))
#         if nb_labels == 2:
#             splits = [[trainIndices for trainIndices, _ in splits],
#                       [testIndices for _, testIndices in splits],
#                       [[] for _ in splits]]
#             return [labels], [(0, 1)], [splits]
#         else:
#             combinations = itertools.combinations(np.arange(nb_labels), 2)
#             multiclass_labels = []
#             labels_indices = []
#             indices_multiclass = []
#             for combination in combinations:
#                 labels_indices.append(combination)
#                 old_indices = [example_index
#                               for example_index, example_label in
#                               enumerate(labels)
#                               if example_label in combination]
#                 train_indices = [np.array([old_index for old_index in old_indices if
#                                           old_index in iterIndices[0]])
#                                 for iterIndices in splits]
#                 test_indices = [np.array([old_index for old_index in old_indices if
#                                          old_index in iterindices[1]])
#                                for iterindices in splits]
#                 test_indices_multiclass = [np.array(iterindices[1]) for
#                                          iterindices in splits]
#                 indices_multiclass.append(
#                     [train_indices, test_indices, test_indices_multiclass])
#                 new_labels = np.zeros(len(labels), dtype=int) - 100
#                 for labelIndex, label in enumerate(labels):
#                     if label == combination[0]:
#                         new_labels[labelIndex] = 1
#                     elif label == combination[1]:
#                         new_labels[labelIndex] = 0
#                     else:
#                         pass
#                 multiclass_labels.append(new_labels)
#
#     elif multiclass_method == "oneVersusRest":
#         # TODO : Implement one versus rest if probas are not a problem anymore
#         pass
#     return multiclass_labels, labels_indices, indices_multiclass


# def gen_multiclass_monoview_decision(monoview_result, classification_indices):
#     learning_indices, validation_indices, test_indices_multiclass = classification_indices
#     multiclass_monoview_decisions = monoview_result.full_labels_pred
#     multiclass_monoview_decisions[
#         test_indices_multiclass] = monoview_result.y_test_multiclass_pred
#     return multiclass_monoview_decisions
#
#
# def is_biclass(multiclass_preds):
#     if multiclass_preds[0] is []:
#         return True
#     else:
#         return False


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
        return self.estimator.get_config()

    def get_interpretation(self, output_file_name=None, y_test=None):
        # return self.estimator.get_interpretation(output_file_name, y_test,
        #                                     multi_class=True)
        # TODO : Multiclass interpretation
        return "Multiclass wrapper is not interpretable yet"


class MonoviewWrapper(MultiClassWrapper):
    pass


class OVRWrapper(MonoviewWrapper, OneVsRestClassifier):
    pass


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


# The following code is a mutliview adaptation of sklearns multiclass package

def _multiview_fit_binary(estimator, X, y, train_indices,
                          view_indices, classes=None, ):
    # TODO : Verifications des sklearn
    estimator = clone(estimator)
    estimator.fit(X, y, train_indices=train_indices,
                  view_indices=view_indices)
    return estimator


def _multiview_predict_binary(estimator, X, example_indices, view_indices):
    if is_regressor(estimator):
        return estimator.predict(X, example_indices=example_indices,
                                 view_indices=view_indices)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X, example_indices=example_indices,
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

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        n_samples = len(example_indices)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _multiview_predict_binary(e, X, example_indices,
                                                 view_indices)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[argmaxima]
        else:
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
                                                       example_indices,
                                                       view_indices) > thresh)[
                        0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)


def _multiview_fit_ovo_binary(estimator, X, y, i, j, train_indices,
                              view_indices):
    cond = np.logical_or(y == i, y == j)
    # y = y[cond]
    y_binary = np.empty(y.shape, np.int)
    y_binary[y == i] = 0
    y_binary[y == j] = 1
    indcond = np.arange(X.get_nb_examples())[cond]
    train_indices = np.intersect1d(train_indices, indcond)
    return _multiview_fit_binary(estimator,
                                 X,
                                 y_binary, train_indices, view_indices,
                                 classes=[i, j]), train_indices


class MultiviewOVOWrapper(MultiviewWrapper, OneVsOneClassifier):

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
        train_indices, view_indices = get_examples_views_indices(X,
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
            estimators_indices[1] if self._pairwise else None)

        return self

    def predict(self, X, example_indices=None, view_indices=None):
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
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        Y = self.multiview_decision_function(X, example_indices=example_indices,
                                             view_indices=view_indices)
        if self.n_classes_ == 2:
            return self.classes_[(Y > 0).astype(np.int)]
        return self.classes_[Y.argmax(axis=1)]

    def multiview_decision_function(self, X, example_indices, view_indices):
        # check_is_fitted(self)

        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            # TODO Gram matrix compatibility
            Xs = [X[:, idx] for idx in indices]
        predictions = np.vstack(
            [est.predict(Xi, example_indices=example_indices,
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
