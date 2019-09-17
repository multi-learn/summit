import unittest

import numpy as np

from multiview_platform.MonoMultiViewClassifiers.Multiview.Additions import \
    diversity_utils
from ....MonoMultiViewClassifiers.MultiviewClassifiers.DoubleFaultFusion import \
    DoubleFaultFusionModule


class Test_doubleFaultRatio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.randomState = np.random.RandomState(42)
        cls.allClassifiersNames = [["SCM", "SVM", "DT"], ["SCM", "SVM", "DT"]]
        cls.directory = ""
        cls.viewsIndices = np.array([0, 1])
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
        cls.folds_ground_truth = np.array([np.array([1,1,1,0,0,0]) for _ in range(3)])

    def test_simple(cls):
        bestCombi, disagreement = diversity_utils.couple_div_measure(
            cls.allClassifiersNames,cls.classifiersDecisions,
            DoubleFaultFusionModule.doubleFault, cls.folds_ground_truth)
        cls.assertAlmostEqual(disagreement, 0.3888888888888)
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
            DoubleFaultFusionModule.doubleFault, cls.folds_ground_truth)
        cls.assertAlmostEqual(disagreement, 0.3333333333)
        cls.assertEqual(len(bestCombi), 3)


class Test_doubleFault(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monoviewDecision1 = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        cls.monoviewDecision2 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        cls.ground_truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    def test_simple(cls):
        disagreement = DoubleFaultFusionModule.doubleFault(
            cls.monoviewDecision1, cls.monoviewDecision2, cls.ground_truth)
        np.testing.assert_array_equal(disagreement, np.array(
            [False, False, False, True, True, False, False, False]))
