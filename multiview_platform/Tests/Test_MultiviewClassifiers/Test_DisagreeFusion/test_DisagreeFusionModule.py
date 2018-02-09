# import unittest
# import numpy as np
#
# from ....MonoMultiViewClassifiers.MultiviewClassifiers.DisagreeFusion import DisagreeFusionModule
#
#
# class Test_disagreement(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.randomState = np.random.RandomState(42)
#         cls.allClassifiersNames = ["SCM", "SVM", "DT"]
#         cls.directory = ""
#         cls.viewsIndices = [0,1]
#         cls.resultsMonoview = [[0, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [0, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [0, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1,6)])
#                                     ]],
#                                [1, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [1, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [1, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0,1,6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6)])
#                                     ]]
#                                ]
#         cls.classificationIndices = []
#
#     def test_simple(cls):
#         bestCombi, disagreement = DisagreeFusionModule.disagree(cls.allClassifiersNames, cls.viewsIndices, cls.resultsMonoview)
#         cls.assertAlmostEqual(disagreement, 0.666666666667)
#         cls.assertEqual(len(bestCombi), 2)
#
#     def test_viewsIndices(cls):
#         cls.viewsIndices = [0,6]
#         cls.resultsMonoview = [[0, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [0, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [0, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [6, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [6, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [6, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6)])
#                                     ]]
#                                ]
#         bestCombi, disagreement = DisagreeFusionModule.disagree(cls.allClassifiersNames, cls.viewsIndices, cls.resultsMonoview)
#         cls.assertAlmostEqual(disagreement, 0.611111111111)
#         cls.assertEqual(len(bestCombi), 2)
#
#     def test_multipleViews(cls):
#         cls.viewsIndices = [0, 6, 18]
#         cls.resultsMonoview = [[0, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [0, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [0, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [6, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [6, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [6, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [18, ["SCM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [18, ["SVM", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6),
#                                                                          cls.randomState.random_integers(0, 1, 6)])
#                                     ]],
#                                [18, ["DT", "", "", "", "", "", np.array([cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6),
#                                                                         cls.randomState.random_integers(0, 1, 6)])
#                                     ]]
#                                ]
#         bestCombi, disagreement = DisagreeFusionModule.disagree(cls.allClassifiersNames, cls.viewsIndices, cls.resultsMonoview,)
#         cls.assertAlmostEqual(disagreement, 0.592592592593)
#         cls.assertEqual(len(bestCombi), 3)