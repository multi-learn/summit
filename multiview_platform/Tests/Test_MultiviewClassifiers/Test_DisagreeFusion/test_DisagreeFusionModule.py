import unittest

import numpy as np

from multiview_platform.MonoMultiViewClassifiers.Multiview.Additions import \
    diversity_utils
from ....MonoMultiViewClassifiers.MultiviewClassifiers.DisagreeFusion import \
    DisagreeFusionModule


class Test_disagreement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.randomState = np.random.RandomState(42)
        cls.allClassifiersNames = ["SCM", "SVM", "DT"]
        cls.directory = ""
        cls.viewsIndices =  np.array([0, 1])
        cls.resultsMonoview =  np.array([[0, ["SCM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [0, ["SVM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [0, ["DT", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [1, ["SCM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [1, ["SVM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [1, ["DT", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]]
                               ])
        cls.classificationIndices = np.array([])

    def test_simple(cls):
        bestCombi, disagreement = diversity_utils.couple_div_measure(
            cls.allClassifiersNames, cls.viewsIndices, cls.resultsMonoview,
            DisagreeFusionModule.disagree)
        cls.assertAlmostEqual(disagreement, 0.666666666667)
        cls.assertEqual(len(bestCombi), 2)

    def test_viewsIndices(cls):
        cls.viewsIndices = np.array([0, 6])
        cls.resultsMonoview =np.array( [[0, ["SCM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [0, ["SVM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [0, ["DT", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [6, ["SCM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [6, ["SVM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [6, ["DT", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]]
                               ])
        bestCombi, disagreement = diversity_utils.couple_div_measure(
            cls.allClassifiersNames, cls.viewsIndices,
            cls.resultsMonoview, DisagreeFusionModule.disagree)
        cls.assertAlmostEqual(disagreement, 0.611111111111)
        cls.assertEqual(len(bestCombi), 2)

    def test_multipleViews(cls):
        cls.viewsIndices = np.array([0, 6, 18])
        cls.resultsMonoview = np.array([[0, ["SCM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [0, ["SVM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [0, ["DT", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [6, ["SCM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [6, ["SVM", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [6, ["DT", "", "", "", "", "",
                                    np.array([cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6),
                                              cls.randomState.randint(0, 2, 6)])
                                    ]],
                               [18, ["SCM", "", "", "", "", "",
                                     np.array([cls.randomState.randint(0, 2, 6),
                                               cls.randomState.randint(0, 2, 6),
                                               cls.randomState.randint(0, 2,
                                                                       6)])
                                     ]],
                               [18, ["SVM", "", "", "", "", "",
                                     np.array([cls.randomState.randint(0, 2, 6),
                                               cls.randomState.randint(0, 2, 6),
                                               cls.randomState.randint(0, 2,
                                                                       6)])
                                     ]],
                               [18, ["DT", "", "", "", "", "",
                                     np.array([cls.randomState.randint(0, 2, 6),
                                               cls.randomState.randint(0, 2, 6),
                                               cls.randomState.randint(0, 2,
                                                                       6)])
                                     ]]
                               ])
        bestCombi, disagreement = diversity_utils.couple_div_measure(
            cls.allClassifiersNames, cls.viewsIndices,
            cls.resultsMonoview, DisagreeFusionModule.disagree)
        cls.assertAlmostEqual(disagreement, 0.592592592593)
        cls.assertEqual(len(bestCombi), 3)


class Test_disagree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.monoviewDecision1 = np.array([0, 0, 1, 1])
        cls.monoviewDecision2 = np.array([0, 1, 0, 1])
        cls.ground_truth = None

    def test_simple(cls):
        disagreement = DisagreeFusionModule.disagree(cls.monoviewDecision1,
                                                     cls.monoviewDecision2,
                                                     cls.ground_truth)
        np.testing.assert_array_equal(disagreement,
                                      np.array([False, True, True, False]))
