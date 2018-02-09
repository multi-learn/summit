import unittest
import numpy as np

from ....MonoMultiViewClassifiers.MultiviewClassifiers.DoubleFaultFusion import DoubleFaultFusionModule
from ....MonoMultiViewClassifiers.MultiviewClassifiers import diversity_utils

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
        cls.ground_truth = np.array([1,1,1,0,0,0])

    def test_simple(cls):
        bestCombi, disagreement = diversity_utils.couple_div_measure(cls.allClassifiersNames, cls.viewsIndices,
                                                                     cls.resultsMonoview,
                                                                     DoubleFaultFusionModule.doubleFault,
                                                                     cls.ground_truth)
        cls.assertAlmostEqual(disagreement, 0.55555555555555547)
        cls.assertEqual(len(bestCombi), 2)

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
        bestCombi, disagreement = diversity_utils.couple_div_measure(cls.allClassifiersNames, cls.viewsIndices,
                                                                     cls.resultsMonoview, DoubleFaultFusionModule.doubleFault,
                                                                     cls.ground_truth)
        cls.assertAlmostEqual(disagreement, 0.33333333333333331)
        cls.assertEqual(len(bestCombi), 2)

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
        bestCombi, disagreement = diversity_utils.couple_div_measure(cls.allClassifiersNames, cls.viewsIndices,
                                                                     cls.resultsMonoview, DoubleFaultFusionModule.doubleFault,
                                                                     cls.ground_truth)
        cls.assertAlmostEqual(disagreement, 0.31481481481481483)
        cls.assertEqual(len(bestCombi), 3)


class Test_doubleFault(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monoviewDecision1 = np.array([0,0,1,1,0,0,1,1])
        cls.monoviewDecision2 = np.array([0,1,0,1,0,1,0,1])
        cls.ground_truth = np.array([0,0,0,0,1,1,1,1])

    def test_simple(cls):
        disagreement = DoubleFaultFusionModule.doubleFault(cls.monoviewDecision1, cls.monoviewDecision2, cls.ground_truth)
        np.testing.assert_array_equal(disagreement, np.array([False,False,False,True,True,False,False,False]))
