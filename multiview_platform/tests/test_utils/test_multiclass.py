import unittest

import numpy as np

from ...mono_multi_view_classifiers.utils import multiclass


class Test_genMulticlassLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.labels = cls.random_state.randint(0, 5, 50)
        cls.testIndices = [
            cls.random_state.choice(np.arange(50), size=10, replace=False),
            cls.random_state.choice(np.arange(50), size=10, replace=False)]
        cls.classification_indices = [
            [np.array([_ for _ in range(50) if _ not in cls.testIndices[0]]),
             cls.testIndices[0]],
            [np.array([_ for _ in range(50) if _ not in cls.testIndices[1]]),
             cls.testIndices[1]]]

    def test_one_versus_one(cls):
        multiclassLabels, labelsIndices, oldIndicesMulticlass = multiclass.genMulticlassLabels(
            cls.labels, "oneVersusOne", cls.classification_indices)
        cls.assertEqual(len(multiclassLabels), 10)
        cls.assertEqual(labelsIndices,
                        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4),
                         (2, 3), (2, 4), (3, 4)])
        np.testing.assert_array_equal(oldIndicesMulticlass[0][0][0],
                                      np.array(
                                          [5, 13, 15, 18, 20, 24, 27, 39, 41,
                                           43, 44, 45, 46, 48]))
        np.testing.assert_array_equal(multiclassLabels[0],
                                      np.array([-100, -100, -100, -100, -100, 0,
                                                -100, -100, -100, -100, -100,
                                                -100,
                                                -100, 0, -100, 0, -100, -100, 1,
                                                -100, 0, -100, -100, 1, 1, -100,
                                                -100,
                                                0, -100, -100, -100, -100, -100,
                                                1, -100, -100, -100, -100, 1, 0,
                                                -100,
                                                1, -100, 0, 0, 1, 0, -100, 0,
                                                -100]))
