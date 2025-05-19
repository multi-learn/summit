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

from summit.multiview_platform.multiview_classifiers import entropy_fusion


class Test_difficulty_fusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.classifiers_decisions = cls.random_state.randint(
            0, 2, size=(5, 3, 5))
        cls.combination = [1, 3, 4]
        cls.y = np.array([1, 1, 0, 0, 1])
        cls.clf = entropy_fusion.EntropyFusion()

    def test_simple(cls):
        entropy = cls.clf.diversity_measure(
            cls.classifiers_decisions,
            cls.combination,
            cls.y)
        cls.assertAlmostEqual(entropy, 0.2)
