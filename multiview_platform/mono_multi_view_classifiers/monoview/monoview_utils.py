import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import uniform, randint

from ..utils.base import BaseClassifier

# Author-Info
__author__ = "Nikolas Huelsmann, Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


# __date__ = 2016 - 03 - 25

def change_label_to_minus(y):
    """
    Change the label 0 to minus one

    Parameters
    ----------
    y :

    Returns
    -------
    label y with -1 instead of 0

    """
    minus_y = np.copy(y)
    minus_y[np.where(y == 0)] = -1
    return minus_y


def change_label_to_zero(y):
    """
    Change the label -1 to 0

    Parameters
    ----------
    y

    Returns
    -------

    """
    zeroed_y = np.copy(y)
    zeroed_y[np.where(y == -1)] = 0
    return zeroed_y


def compute_possible_combinations(params_dict):
    n_possibs = np.ones(len(params_dict)) * np.inf
    for value_index, value in enumerate(params_dict.values()):
        if type(value) == list:
            n_possibs[value_index] = len(value)
        elif isinstance(value, CustomRandint):
            n_possibs[value_index] = value.get_nb_possibilities()
    return n_possibs


def gen_test_folds_preds(X_train, y_train, KFolds, estimator):
    test_folds_preds = []
    train_index = np.arange(len(y_train))
    folds = KFolds.split(train_index, y_train)
    fold_lengths = np.zeros(KFolds.n_splits, dtype=int)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        fold_lengths[fold_index] = len(test_indices)
        estimator.fit(X_train[train_indices], y_train[train_indices])
        test_folds_preds.append(estimator.predict(X_train[train_indices]))
    min_fold_length = fold_lengths.min()
    test_folds_preds = np.array(
        [test_fold_preds[:min_fold_length] for test_fold_preds in
         test_folds_preds])
    return test_folds_preds


class CustomRandint:
    """Used as a distribution returning a integer between low and high-1.
    It can be used with a multiplier agrument to be able to perform more complex generation
    for example 10 e -(randint)"""

    def __init__(self, low=0, high=0, multiplier=""):
        self.randint = randint(low, high)
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        randinteger = self.randint.rvs(random_state=random_state)
        if self.multiplier == "e-":
            return 10 ** -randinteger
        else:
            return randinteger

    def get_nb_possibilities(self):
        return self.randint.b - self.randint.a


class CustomUniform:
    """Used as a distribution returning a float between loc and loc + scale..
        It can be used with a multiplier agrument to be able to perform more complex generation
        for example 10 e -(float)"""

    def __init__(self, loc=0, state=1, multiplier=""):
        self.uniform = uniform(loc, state)
        self.multiplier = multiplier

    def rvs(self, random_state=None):
        unif = self.uniform.rvs(random_state=random_state)
        if self.multiplier == 'e-':
            return 10 ** -unif
        else:
            return unif


class BaseMonoviewClassifier(BaseClassifier):

    def get_config(self):
        if self.param_names:
            return "\n\t\t- " + self.__class__.__name__ + "with " + self.params_to_string()
        else:
            return "\n\t\t- " + self.__class__.__name__ + "with no config."

    def get_feature_importance(self, directory, nb_considered_feats=50):
        """Used to generate a graph and a pickle dictionary representing feature importances"""
        feature_importances = self.feature_importances_
        sorted_args = np.argsort(-feature_importances)
        feature_importances_sorted = feature_importances[sorted_args][
                                     :nb_considered_feats]
        feature_indices_sorted = sorted_args[:nb_considered_feats]
        fig, ax = plt.subplots()
        x = np.arange(len(feature_indices_sorted))
        formatter = FuncFormatter(percent)
        ax.yaxis.set_major_formatter(formatter)
        plt.bar(x, feature_importances_sorted)
        plt.title("Importance depending on feature")
        fig.savefig(directory + "feature_importances.png", transparent=True)
        plt.close()
        features_importances_dict = dict((featureIndex, featureImportance)
                                         for featureIndex, featureImportance in
                                         enumerate(feature_importances)
                                         if featureImportance != 0)
        with open(directory + 'feature_importances.pickle', 'wb') as handle:
            pickle.dump(features_importances_dict, handle)
        interpret_string = "Feature importances : \n"
        for featureIndex, featureImportance in zip(feature_indices_sorted,
                                                   feature_importances_sorted):
            if featureImportance > 0:
                interpret_string += "- Feature index : " + str(featureIndex) + \
                                    ", feature importance : " + str(
                    featureImportance) + "\n"
        return interpret_string

    def get_name_for_fusion(self):
        return self.__class__.__name__[:4]


def percent(x, pos):
    """Used to print percentage of importance on the y axis"""
    return '%1.1f %%' % (x * 100)


class MonoviewResult(object):
    def __init__(self, view_index, classifier_name, view_name, metrics_scores,
                 full_labels_pred, classifier_config, test_folds_preds,
                 classifier, n_features):
        self.view_index = view_index
        self.classifier_name = classifier_name
        self.view_name = view_name
        self.metrics_scores = metrics_scores
        self.full_labels_pred = full_labels_pred
        self.classifier_config = classifier_config
        self.test_folds_preds = test_folds_preds
        self.clf = classifier
        self.n_features = n_features

    def get_classifier_name(self):
        return self.classifier_name + "-" + self.view_name


def get_accuracy_graph(plotted_data, classifier_name, file_name,
                       name="Accuracies", bounds=None, bound_name=None,
                       boosting_bound=None, set="train", zero_to_one=True):
    if type(name) is not str:
        name = " ".join(name.getConfig().strip().split(" ")[:2])
    f, ax = plt.subplots(nrows=1, ncols=1)
    if zero_to_one:
        ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_title(name + " during " + set + " for " + classifier_name)
    x = np.arange(len(plotted_data))
    scat = ax.scatter(x, np.array(plotted_data), marker=".")
    if bounds:
        if boosting_bound:
            scat2 = ax.scatter(x, boosting_bound, marker=".")
            scat3 = ax.scatter(x, np.array(bounds), marker=".", )
            ax.legend((scat, scat2, scat3),
                      (name, "Boosting bound", bound_name))
        else:
            scat2 = ax.scatter(x, np.array(bounds), marker=".", )
            ax.legend((scat, scat2),
                      (name, bound_name))
        # plt.tight_layout()
    else:
        ax.legend((scat,), (name,))
    f.savefig(file_name, transparent=True)
    plt.close()
