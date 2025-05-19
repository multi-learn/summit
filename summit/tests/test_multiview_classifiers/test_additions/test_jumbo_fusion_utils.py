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
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
# Version:
# -------
#
# * multiview_generator version = 0.0.1
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
#
import unittest
import numpy as np

import summit.multiview_platform.multiview_classifiers.additions.jumbo_fusion_utils as ju


class FakeDataset():

    def __init__(self, views, labels):
        self.nb_views = views.shape[0]
        self.dataset_length = views.shape[2]
        self.views = views
        self.labels = labels

    def get_v(self, view_index, sample_indices):
        return self.views[view_index, sample_indices]

    def get_nb_class(self, sample_indices):
        return np.unique(self.labels[sample_indices])


# TODO
