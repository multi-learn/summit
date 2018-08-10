import unittest
import numpy as np

from ....MonoMultiViewClassifiers.MultiviewClassifiers.DifficultyFusion import DifficultyFusionModule

class Test_difficulty(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifiersDecisions = np.array([
            [np.random.randint(0,2, (2, 5)), np.array([[0, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
             np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)),
             np.random.randint(0,2, (2, 5))],
            [np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)),
             np.random.randint(0,2, (2, 5)), np.array([[0, 0, 1, 1, 0], [0, 1, 0, 1, 0]]),
             np.random.randint(0,2, (2, 5))],
            [np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)),
             np.random.randint(0,2, (2, 5)), np.random.randint(0,2, (2, 5)),
             np.array([[0, 1, 1, 1, 1], [0, 1, 0, 1, 0]])],
        ])
        cls.combination = [1, 3, 4]
        cls.foldsGroudTruth = np.array([[1, 1, 0, 0, 1], [0, 1, 0, 1, 0]])
        cls.foldsLen = ""

    def test_simple(cls):
        difficulty_measure = DifficultyFusionModule.difficulty(cls.classifiersDecisions,
                                                               cls.combination,
                                                               cls.foldsGroudTruth,
                                                               cls.foldsLen)
        cls.assertAlmostEqual(difficulty_measure, 0.29861111111)
