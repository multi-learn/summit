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
