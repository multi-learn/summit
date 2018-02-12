import unittest
import numpy as np

from ...MonoMultiViewClassifiers.MultiviewClassifiers import diversity_utils


def fake_measure(a, b, c, d, e):
    return 42


class Test_global_div_measure(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.allClassifersNames = ["SCM", "DT", "SVM"]
        cls.viewsIndices = [0,1]
        cls.randomState = np.random.RandomState(42)
        cls.resultsMonoview = [[0, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [0, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [0, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1,6)])
                                    ]],
                               [1, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [1, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [1, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6)])
                                    ]]
                               ]
        cls.measurement = fake_measure
        cls.foldsGroudTruth = np.array([cls.randomState.random_integers(0, 1, 6),
                                        cls.randomState.random_integers(0, 1, 6),
                                        cls.randomState.random_integers(0, 1, 6)])

    def test_simple(cls):
        clf_names, diversity_measure = diversity_utils.global_div_measure(cls.allClassifersNames,
                                                                          cls.viewsIndices,
                                                                          cls.resultsMonoview,
                                                                          cls.measurement,
                                                                          cls.foldsGroudTruth)
        cls.assertEqual(len(clf_names), 2)
        cls.assertEqual(diversity_measure, 42)