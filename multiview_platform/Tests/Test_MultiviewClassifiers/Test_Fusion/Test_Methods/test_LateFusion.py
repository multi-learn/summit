import unittest
import numpy as np

from .....MonoMultiViewClassifiers.MultiviewClassifiers.Fusion.Methods import LateFusion

class Test_disagreement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.randomState = np.random.RandomState(42)
        cls.allClassifiersNames = ["SCM", "SVM", "DT"]
        cls.directory = ""
        cls.viewsIndices = [0,1]
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
        cls.classificationIndices = []

    def test_simple(cls):
        bestCombi = LateFusion.disagreement(cls.allClassifiersNames, cls.directory, cls.viewsIndices,
                                            cls.resultsMonoview, cls.classificationIndices)
        cls.assertEqual(bestCombi, ["SCM", "DT"])

    def test_viewsIndices(cls):
        cls.viewsIndices = [0,6]
        cls.resultsMonoview = [[0, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [0, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [0, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [6, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [6, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [6, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6)])
                                    ]]
                               ]
        bestCombi = LateFusion.disagreement(cls.allClassifiersNames, cls.directory, cls.viewsIndices,
                                            cls.resultsMonoview, cls.classificationIndices)
        cls.assertEqual(bestCombi, ["DT", "DT"])

    def test_multipleViews(cls):
        cls.viewsIndices = [0, 6, 18]
        cls.resultsMonoview = [[0, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [0, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [0, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [6, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [6, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [6, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [18, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [18, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6),
                                                                         cls.randomState.random_integers(0, 1, 6)])
                                    ]],
                               [18, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6),
                                                                        cls.randomState.random_integers(0, 1, 6)])
                                    ]]
                               ]
        bestCombi = LateFusion.disagreement(cls.allClassifiersNames, cls.directory, cls.viewsIndices,
                                            cls.resultsMonoview, cls.classificationIndices)
        cls.assertEqual(bestCombi, ['SCM', 'SVM', 'SVM'])