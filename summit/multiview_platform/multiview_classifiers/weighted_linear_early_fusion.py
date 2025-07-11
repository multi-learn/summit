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

from .additions.fusion_utils import BaseFusionClassifier
from ..multiview.multiview_utils import get_available_monoview_classifiers, \
    BaseMultiviewClassifier, ConfigGenerator
from ..utils.dataset import get_samples_views_indices
from ..utils.multiclass import get_mc_estim, MultiClassWrapper

# from ..utils.dataset import get_v

classifier_class_name = "WeightedLinearEarlyFusion"


class WeightedLinearEarlyFusion(BaseMultiviewClassifier, BaseFusionClassifier):
    """
    Builds a monoview dataset by concatenating the views (with a weight if
    needed) and learns a monoview classifier on the concatenation
    """

    def __init__(self, random_state=None, view_weights=None,
                 monoview_classifier_name="decision_tree",
                 monoview_classifier_config={}):
        BaseMultiviewClassifier.__init__(self, random_state=random_state)
        self.view_weights = view_weights
        self.monoview_classifier_name = monoview_classifier_name
        self.short_name = "early_fusion"
        if monoview_classifier_name in monoview_classifier_config:
            self.monoview_classifier_config = monoview_classifier_config[
                monoview_classifier_name]
        self.monoview_classifier_config = monoview_classifier_config
        self.monoview_classifier = self.init_monoview_estimator(
            monoview_classifier_name, monoview_classifier_config)
        self.param_names = ["monoview_classifier_name",
                            "monoview_classifier_config"]
        self.distribs = [get_available_monoview_classifiers(),
                         ConfigGenerator(get_available_monoview_classifiers())]
        self.classed_params = []
        self.weird_strings = {}

    def set_params(self, monoview_classifier_name="decision_tree",
                   monoview_classifier_config={}, **params):
        self.monoview_classifier_name = monoview_classifier_name
        self.monoview_classifier = self.init_monoview_estimator(
            monoview_classifier_name,
            monoview_classifier_config)
        self.monoview_classifier_config = self.monoview_classifier.get_params()
        self.short_name = "early_fusion"
        return self

    def get_params(self, deep=True):
        return {"random_state": self.random_state,
                "view_weights": self.view_weights,
                "monoview_classifier_name": self.monoview_classifier_name,
                "monoview_classifier_config": self.monoview_classifier_config}

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, X = self.transform_data_to_monoview(X, train_indices,
                                                           view_indices)
        self.used_views = view_indices
        if np.unique(y[train_indices]).shape[0] > 2 and \
                not (isinstance(self.monoview_classifier, MultiClassWrapper)):
            self.monoview_classifier = get_mc_estim(self.monoview_classifier,
                                                    self.random_state,
                                                    multiview=False,
                                                    y=y[train_indices])
        self.monoview_classifier.fit(X, y[train_indices])
        self.monoview_classifier_config = self.monoview_classifier.get_params()
        if hasattr(self.monoview_classifier, 'feature_importances_'):
            self.feature_importances_ = self.monoview_classifier.feature_importances_
        return self

    def predict(self, X, sample_indices=None, view_indices=None):
        _, X = self.transform_data_to_monoview(X, sample_indices, view_indices)
        self._check_views(self.view_indices)
        predicted_labels = self.monoview_classifier.predict(X)
        return predicted_labels

    def transform_data_to_monoview(self, dataset, sample_indices,
                                   view_indices):
        """Here, we extract the data from the HDF5 dataset file and store all
        the concatenated views in one variable"""
        sample_indices, self.view_indices = get_samples_views_indices(dataset,
                                                                      sample_indices,
                                                                      view_indices)
        if self.view_weights is None:
            self.view_weights = np.ones(len(self.view_indices), dtype=float)
        else:
            self.view_weights = np.array(self.view_weights)
        self.view_weights /= float(np.sum(self.view_weights))

        X = self.hdf5_to_monoview(dataset, sample_indices)
        return sample_indices, X

    def hdf5_to_monoview(self, dataset, samples):
        """Here, we concatenate the views for the asked samples """
        monoview_data = np.concatenate(
            [dataset.get_v(view_idx, samples)
             for view_weight, (index, view_idx)
             in zip(self.view_weights, enumerate(self.view_indices))], axis=1)
        return monoview_data

    # def set_monoview_classifier_config(self, monoview_classifier_name, monoview_classifier_config):
    #     if monoview_classifier_name in monoview_classifier_config:
    #         self.monoview_classifier.set_params(**monoview_classifier_config[monoview_classifier_name])
    #     else:
    #         self.monoview_classifier.set_params(**monoview_classifier_config)
