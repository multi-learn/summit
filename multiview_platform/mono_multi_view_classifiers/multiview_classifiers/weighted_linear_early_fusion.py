import numpy as np

from ..utils.dataset import getV
from .additions.utils import BaseMultiviewClassifier, get_train_views_indices
from .. import monoview_classifiers


class WeightedLinearEarlyFusion(BaseMultiviewClassifier):

    def __init__(self, view_weights=None, monoview_classifier="decision_tree", monoview_classifier_config=None, random_state=42):
        super(WeightedLinearEarlyFusion, self).__init__(random_state=random_state)
        self.view_weights = np.array(view_weights)
        if type(monoview_classifier) == str:
            monoview_classifier_module = getattr(monoview_classifiers,
                                               monoview_classifier)
            monoview_classifier_class = getattr(monoview_classifier_module, monoview_classifier_module.classifier_class_name)
            self.monoview_classifier = monoview_classifier_class(**monoview_classifier_config)
        else:
            self.monoview_classifier = monoview_classifier

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, X = self.transform_data_to_monoview(X, train_indices, view_indices)
        self.monoview_classifier.fit(X, y[train_indices])

    def predict(self, X, predict_indices, view_indices):
        _, X = self.transform_data_to_monoview(X, predict_indices, view_indices)
        predicted_labels = self.monoview_classifier.predict(X)
        return predicted_labels

    def transform_data_to_monoview(self, dataset, example_indices, view_indices):
        """Here, we extract the data from the HDF5 dataset file and store all
        the concatenated views in one variable"""
        example_indices, self.view_indices = get_train_views_indices(dataset,
                                                                        example_indices,
                                                                        view_indices)
        if self.view_weights is None:
            self.view_weights = np.ones(len(self.view_indices), dtype=float)
        self.view_weights /= float(np.sum(self.view_weights))

        X = self.hdf5_to_monoview(dataset, example_indices, self.view_indices)
        return example_indices, X

    def hdf5_to_monoview(self, dataset, exmaples, view_indices):
        """Here, we concatenate the views for the asked examples """
        monoview_data = np.concatenate(
            [getV(dataset, view_idx, exmaples)
             for view_weight, (index, view_idx)
             in zip(self.view_weights, enumerate(view_indices))]
            , axis=1)
        return monoview_data





