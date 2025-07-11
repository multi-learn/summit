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

from ..multiview_classifiers.additions.late_fusion_utils import \
    LateFusionClassifier
from ..utils.dataset import get_samples_views_indices

classifier_class_name = "WeightedLinearLateFusion"


class WeightedLinearLateFusion(LateFusionClassifier):

    """
    Similar to the majority voting fusion.
    """
    def __init__(self, random_state, classifiers_names=None,
                 classifier_configs=None, weights=None, nb_cores=1, rs=None):
        self.need_probas = True
        LateFusionClassifier.__init__(self, random_state=random_state,
                                      classifiers_names=classifiers_names,
                                      classifier_configs=classifier_configs,
                                      nb_cores=nb_cores, weights=weights, rs=rs)

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        view_scores = []
        for index, viewIndex in enumerate(view_indices):
            view_scores.append(
                np.array(self.monoview_estimators[index].predict_proba(
                    X.get_v(viewIndex, sample_indices))) * self.weights[index])
        view_scores = np.array(view_scores)
        predicted_labels = np.argmax(np.sum(view_scores, axis=0), axis=1)
        return predicted_labels
