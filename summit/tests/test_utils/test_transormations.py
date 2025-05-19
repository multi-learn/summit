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


from summit.multiview_platform.utils import transformations


class TestFunctions(unittest.TestCase):

    def test_simple_sign(self):
        trans = transformations.sign_labels(np.zeros(10))
        np.testing.assert_array_equal(np.ones(10) * -1, trans)
        trans = transformations.sign_labels(np.ones(10))
        np.testing.assert_array_equal(np.ones(10), trans)

    def test_simple_unsign(self):
        trans = transformations.unsign_labels(np.ones(10) * -1)
        np.testing.assert_array_equal(np.zeros(10), trans)
        trans = transformations.unsign_labels(np.ones(10).reshape((10, 1)))
        np.testing.assert_array_equal(np.ones(10), trans)
