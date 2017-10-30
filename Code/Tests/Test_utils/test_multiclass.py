import unittest
import numpy as np

from ...MonoMultiViewClassifiers.utils import Multiclass


class Test_genMulticlassLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.labels = cls.random_state.randint(0,5,50)

    def test_one_versus_one(cls):
        multiclassLabels, labelsIndices, oldIndicesMulticlass = Multiclass.genMulticlassLabels(cls.labels, "oneVersusOne")
        cls.assertEqual(len(multiclassLabels), 10)
        cls.assertEqual(labelsIndices, [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)])
        np.testing.assert_array_equal(oldIndicesMulticlass[0],
                                      np.array([5, 13, 15, 18, 20, 23, 24, 27, 33, 38, 39, 41, 43, 44, 45, 46, 48]))
        np.testing.assert_array_equal(multiclassLabels[0],
                                      np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]))
