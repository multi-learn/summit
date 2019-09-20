import unittest

import numpy as np

from multiview_platform.mono_multi_view_classifiers.multiview.additions import \
    diversity_utils
from ....mono_multi_view_classifiers.multiview_classifiers.disagree_fusion import \
    disagree_fusion
from multiview_platform.mono_multi_view_classifiers.multiview.multiview_utils import MultiviewResult

class Test_disagreement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.randomState = np.random.RandomState(42)
        cls.allClassifiersNames = [["SCM", "SVM", "DT"], ["SCM", "SVM", "DT"]]
        cls.viewsIndices =  np.array([0, 1])
        cls.classifiersDecisions = np.zeros((cls.viewsIndices.shape[0], len(cls.allClassifiersNames), 3, 6),
                                            dtype=int)
        for classifer_index, classifier in enumerate(cls.allClassifiersNames):
            for view_index, view in enumerate(cls.viewsIndices):
                cls.classifiersDecisions[view_index, classifer_index] = np.array([
                    cls.randomState.randint(0, 2, 6),
                    cls.randomState.randint(0, 2, 6),
                    cls.randomState.randint(0, 2, 6)])
        cls.folds_ground_truth = np.array([np.array([1,1,1,0,0,0]) for _ in range(3)])
        cls.classificationIndices = np.array([])

    def test_simple(cls):
        bestCombi, disagreement = diversity_utils.couple_div_measure(
            cls.allClassifiersNames, cls.classifiersDecisions, disagree_fusion.disagree, cls.folds_ground_truth)
        cls.assertAlmostEqual(disagreement, 0.666666666667)
        cls.assertEqual(len(bestCombi), 2)

    def test_multipleViews(cls):
        cls.viewsIndices = np.array([0, 6, 18])
        cls.allClassifiersNames = [["SCM", "SVM", "DT"], ["SCM", "SVM", "DT"], ["SCM", "SVM", "DT"]]
        cls.classifiersDecisions = np.zeros(
            (cls.viewsIndices.shape[0], len(cls.allClassifiersNames), 3, 6),
            dtype=int)
        for classifer_index, classifier in enumerate(cls.allClassifiersNames):
            for view_index, view in enumerate(cls.viewsIndices):
                cls.classifiersDecisions[
                    view_index, classifer_index] = np.array([
                    cls.randomState.randint(0, 2, 6),
                    cls.randomState.randint(0, 2, 6),
                    cls.randomState.randint(0, 2, 6)])

        bestCombi, disagreement = diversity_utils.couple_div_measure(
            cls.allClassifiersNames, cls.classifiersDecisions,
            disagree_fusion.disagree, cls.folds_ground_truth)
        cls.assertAlmostEqual(disagreement, 0.55555555555555)
        cls.assertEqual(len(bestCombi), 3)


class Test_disagree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monoviewDecision1 = np.array([0, 0, 1, 1])
        cls.monoviewDecision2 = np.array([0, 1, 0, 1])
        cls.ground_truth = None

    def test_simple(cls):
        disagreement = disagree_fusion.disagree(cls.monoviewDecision1,
                                                cls.monoviewDecision2,
                                                cls.ground_truth)
        np.testing.assert_array_equal(disagreement,
                                      np.array([False, True, True, False]))
