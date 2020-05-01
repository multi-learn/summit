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
