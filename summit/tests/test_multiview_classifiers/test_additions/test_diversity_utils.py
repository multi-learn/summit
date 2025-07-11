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
import unittest
import numpy as np

import summit.multiview_platform.multiview_classifiers.additions.diversity_utils as du


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


class FakeDivCoupleClf(du.CoupleDiversityFusionClassifier):

    def __init__(self, rs, classifier_names=None,
                 classifiers_config=None, monoview_estimators=None):
        super(FakeDivCoupleClf, self).__init__(random_state=rs,
                                               classifier_names=classifier_names,
                                               classifier_configs=classifiers_config,
                                               monoview_estimators=monoview_estimators)
        self.rs = rs

    def diversity_measure(self, a, b, c):
        return self.rs.randint(0, 100)


class FakeDivGlobalClf(du.GlobalDiversityFusionClassifier):

    def __init__(self, rs, classifier_names=None,
                 classifiers_config=None, monoview_estimators=None):
        super(FakeDivGlobalClf, self).__init__(random_state=rs,
                                               classifier_names=classifier_names,
                                               classifier_configs=classifiers_config,
                                               monoview_estimators=monoview_estimators)
        self.rs = rs

    def diversity_measure(self, a, b, c):
        return self.rs.randint(0, 100)


class Test_DiversityFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier_names = ["adaboost", "decision_tree"]
        cls.classifiers_config = {"adaboost": {"n_estimators": 5, }}
        cls.random_state = np.random.RandomState(42)
        cls.y = cls.random_state.randint(0, 2, 6)
        cls.X = FakeDataset(cls.random_state.randint(0, 100, (2, 5, 6)), cls.y)
        cls.train_indices = [0, 1, 2, 4]
        cls.views_indices = [0, 1]

    def test_simple_couple(self):
        clf = FakeDivCoupleClf(self.random_state, classifier_names=self.classifier_names,
                               classifiers_config=self.classifiers_config)
        clf.fit(self.X, self.y, self.train_indices, self.views_indices)

    def test_simple_global(self):
        clf = FakeDivGlobalClf(self.random_state,
                               classifier_names=self.classifier_names,
                               classifiers_config=self.classifiers_config)
        clf.fit(self.X, self.y, self.train_indices, self.views_indices)


if __name__ == '__main__':
    unittest.main()
