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
# import unittest
#
# import numpy as np
#
# from ....multiview_platform.multiview_classifiers.entropy_fusion_old import EntropyFusionModule
#
# class Test_entropy(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.classifiersDecisions = np.array([
#             [np.random.randint(0,2,(2,5)), [[0,0,1,0,1], [0,1,0,1,0]], np.random.randint(0,2,(2,5)), np.random.randint(0,2,(2,5)), np.random.randint(0,2,(2,5))],
#             [np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)), [[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]], np.random.randint(0,2, (2, 5))],
#             [np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)), [[0, 1, 1, 1, 1], [0, 1, 0, 1, 0]]],
#             ])
#         cls.combination = [1,3,4]
#         cls.foldsGroudTruth = np.array([[1,1,0,0,1], [0,1,0,1,0]])
#         cls.foldsLen = ""
#
#     def test_simple(cls):
#         entropy_score = EntropyFusionModule.entropy(cls.classifiersDecisions, cls.combination, cls.foldsGroudTruth,cls.foldsLen)
#         cls.assertEqual(entropy_score, 0.15, 'Wrong values for entropy measure')
