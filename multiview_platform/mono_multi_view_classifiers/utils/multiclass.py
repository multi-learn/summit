import itertools
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import numpy as np

from .base import BaseClassifier


def gen_multiclass_labels(labels, multiclass_method, splits):
    r"""Used to gen the train/test splits and to set up the framework of the adaptation of a multiclass dataset
    to biclass algorithms.

    First, the function checks whether the dataset is really multiclass.

    Then, it generates all the possible couples of different labels in order to perform one versus one classification.

    For each combination, it selects the examples in the training sets (for each statistical iteration) that have their
    label in the combination and does the same for the testing set. It also saves the multiclass testing set in order to
    use multiclass metrics on the decisions.

    Lastly, it creates a new array of biclass labels (0/1) for the biclass classifications used in oneVersusOne

    Parameters
    ----------
    labels : numpy.ndarray
        Name of the database.
    multiclass_method : string
        The name of the multiclass method used (oneVersusOne, oneVersusAll, ...).
    splits : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.

    Returns
    -------
    multiclass_labels : list of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.

    labels_indices : list of lists of numpy.ndarray
        Each original couple of different labels.

    indices_multiclass : list of lists of numpy.ndarray
        For each combination, contains a biclass labels numpy.ndarray with the 0/1 labels of combination.
    """
    if multiclass_method == "oneVersusOne":
        nb_labels = len(set(list(labels)))
        if nb_labels == 2:
            splits = [[trainIndices for trainIndices, _ in splits],
                      [testIndices for _, testIndices in splits],
                      [[] for _ in splits]]
            return [labels], [(0, 1)], [splits]
        else:
            combinations = itertools.combinations(np.arange(nb_labels), 2)
            multiclass_labels = []
            labels_indices = []
            indices_multiclass = []
            for combination in combinations:
                labels_indices.append(combination)
                old_indices = [example_index
                              for example_index, example_label in
                              enumerate(labels)
                              if example_label in combination]
                train_indices = [np.array([old_index for old_index in old_indices if
                                          old_index in iterIndices[0]])
                                for iterIndices in splits]
                test_indices = [np.array([old_index for old_index in old_indices if
                                         old_index in iterindices[1]])
                               for iterindices in splits]
                test_indices_multiclass = [np.array(iterindices[1]) for
                                         iterindices in splits]
                indices_multiclass.append(
                    [train_indices, test_indices, test_indices_multiclass])
                new_labels = np.zeros(len(labels), dtype=int) - 100
                for labelIndex, label in enumerate(labels):
                    if label == combination[0]:
                        new_labels[labelIndex] = 1
                    elif label == combination[1]:
                        new_labels[labelIndex] = 0
                    else:
                        pass
                multiclass_labels.append(new_labels)

    elif multiclass_method == "oneVersusRest":
        # TODO : Implement one versus rest if probas are not a problem anymore
        pass
    return multiclass_labels, labels_indices, indices_multiclass


def gen_multiclass_monoview_decision(monoview_result, classification_indices):
    learning_indices, validation_indices, test_indices_multiclass = classification_indices
    multiclass_monoview_decisions = monoview_result.full_labels_pred
    multiclass_monoview_decisions[
        test_indices_multiclass] = monoview_result.y_test_multiclass_pred
    return multiclass_monoview_decisions


def is_biclass(multiclass_preds):
    if multiclass_preds[0] is []:
        return True
    else:
        return False


def get_mc_estim(estimator, random_state):
    # print(estimator.accepts_multi_class(random_state))
    if not estimator.accepts_multi_class(random_state):
        if hasattr(estimator, "predict_proba"):
            estimator = OVRWrapper(estimator)
            print(estimator.get_params().keys())
        else:
            estimator = OneVsOneClassifier(estimator)
    return estimator

class MCWrapper():

    def set_params(self, **params):
        self.estimator.set_params(**params)
        return self

    def get_config(self):
        return self.estimator.get_config()

    def get_interpret(self, output_file_name, y_test):
        return self.estimator.get_interpret(output_file_name, y_test,
                                            multi_class=True)

#
#
class OVRWrapper(MCWrapper, OneVsOneClassifier):

    pass


class OVOWrapper(MCWrapper, BaseClassifier):

    pass
