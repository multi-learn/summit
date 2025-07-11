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
# # import unittest
#
import numpy as np
import unittest
#
from summit.multiview_platform.multiview_classifiers import disagree_fusion


class Test_disagree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monoview_decision_1 = np.array([0, 0, 1, 1])
        cls.monoview_decision_2 = np.array([0, 1, 0, 1])
        cls.ground_truth = None
        cls.clf = disagree_fusion.DisagreeFusion()

    def test_simple(cls):
        disagreement = cls.clf.diversity_measure(cls.monoview_decision_1,
                                                 cls.monoview_decision_2,
                                                 cls.ground_truth)
        np.testing.assert_array_equal(disagreement,
                                      np.array([False, True, True, False]))
