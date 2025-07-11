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
import numpy as np

from ... import monoview_classifiers
from ...multiview.multiview_utils import get_available_monoview_classifiers, \
    BaseMultiviewClassifier, ConfigGenerator
from ...utils.dataset import get_samples_views_indices
from ...utils.multiclass import get_mc_estim, MultiClassWrapper

# from ..utils.dataset import get_v

classifier_class_name = "WeightedLinearEarlyFusion"


class BaseEarlyFusion(BaseMultiviewClassifier):

    def __init__(self, monoview_classifier="decision_tree", random_state=None,
                 **kwargs):
        BaseMultiviewClassifier.__init__(self, random_state=random_state)
        monoview_classifier_module = getattr(monoview_classifiers, monoview_classifier)
        monoview_classifier_class = getattr(monoview_classifier_module, monoview_classifier_module.classifier_class_name)
        self.monoview_classifier = monoview_classifier_class(**kwargs)

    def set_params(self, **params):
        self.monoview_classifier.set_params(**params)
        return self

    def get_params(self, deep=True):
        monoview_params = self.monoview_classifier.get_params(deep=deep)
        monoview_params["random_state"] = self.random_state
        return monoview_params

    def fit(self, X, y, train_indices=None, view_indices=None):
        self.view_dict = X.view_dict
        train_indices, self.view_indices = get_samples_views_indices(X,
                                                                      train_indices,
                                                                      view_indices)
        train_indices, X = self.transform_data_to_monoview(X, train_indices,)
        self.used_views = view_indices
        if np.unique(y[train_indices]).shape[0] > 2 and \
                not (isinstance(self.monoview_classifier, MultiClassWrapper)):
            self.monoview_classifier = get_mc_estim(self.monoview_classifier,
                                                    self.random_state,
                                                    multiview=False,
                                                    y=y[train_indices])
        self.monoview_classifier.fit(X, y[train_indices])
        if hasattr(self.monoview_classifier, "feature_importances_"):
            self.get_feature_importances()
        return self

    def predict(self, X, sample_indices=None, view_indices=None):
        _, X = self.transform_data_to_monoview(X, sample_indices)
        self._check_views(self.view_indices)
        predicted_labels = self.monoview_classifier.predict(X)
        return predicted_labels

    def get_feature_importances(self):
        self.feature_importances_ = self.monoview_classifier.feature_importances_

    def transform_data_to_monoview(self, dataset, sample_indices):
        """Here, we extract the data from the HDF5 dataset file and store all
        the concatenated views in one variable"""
        X = self.hdf5_to_monoview(dataset, sample_indices)
        return sample_indices, X

    def hdf5_to_monoview(self, dataset, samples):
        """Here, we concatenate the views for the asked samples """
        monoview_data = np.concatenate(
            [dataset.get_v(view_idx, samples)
             for index, view_idx
             in enumerate(self.view_indices)], axis=1)
        self.feature_ids = []
        for view_idx in self.view_indices:
            view_name = dataset.view_names[view_idx]
            self.feature_ids += [view_name+"-"+feat_id for feat_id in dataset.feature_ids[view_idx]]
        return monoview_data