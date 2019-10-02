# import unittest
# import numpy as np
#
# from ..mono_multi_view_classifiers import ResultAnalysis
#
#
# class Test_getMetricsScoresBiclass(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.metrics = [["accuracy_score"]]
#         cls.monoViewResults = [["", ["chicken_is_heaven", ["View0"], {"accuracy_score": [0.5,0.7]}]]]
#         cls.multiviewResults = [["Mumbo", {"":""}, {"accuracy_score":[0.6,0.8]}]]
#
#     def test_simple(cls):
#         res = ResultAnalysis.getMetricsScoresBiclass(cls.metrics, cls.monoViewResults, cls.multiviewResults)
#         cls.assertIn("accuracy_score",res)
#         cls.assertEqual(type(res["accuracy_score"]), dict)
#         cls.assertEqual(res["accuracy_score"]["classifiers_names"], ["chicken_is_heaven-View0", "Mumbo"])
#         cls.assertEqual(res["accuracy_score"]["train_scores"], [0.5, 0.6])
#         cls.assertEqual(res["accuracy_score"]["test_scores"], [0.7, 0.8])
#
#     def test_only_multiview(cls):
#         cls.monoViewResults = []
#         res = ResultAnalysis.getMetricsScoresBiclass(cls.metrics, cls.monoViewResults, cls.multiviewResults)
#         cls.assertIn("accuracy_score",res)
#         cls.assertEqual(type(res["accuracy_score"]), dict)
#         cls.assertEqual(res["accuracy_score"]["classifiers_names"], ["Mumbo"])
#         cls.assertEqual(res["accuracy_score"]["train_scores"], [0.6])
#         cls.assertEqual(res["accuracy_score"]["test_scores"], [0.8])
#
#     def test_only_monoview(cls):
#         cls.multiviewResults = []
#         res = ResultAnalysis.getMetricsScoresBiclass(cls.metrics, cls.monoViewResults, cls.multiviewResults)
#         cls.assertIn("accuracy_score",res)
#         cls.assertEqual(type(res["accuracy_score"]), dict)
#         cls.assertEqual(res["accuracy_score"]["classifiers_names"], ["chicken_is_heaven-View0"])
#         cls.assertEqual(res["accuracy_score"]["train_scores"], [0.5])
#         cls.assertEqual(res["accuracy_score"]["test_scores"], [0.7])
#
#
# class Test_getExampleErrorsBiclass(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.usedBenchmarkArgumentDictionary = {"labels": np.array([0,1,1,-100,-100,0,1,1,-100])}
#         cls.monoViewResults = [["", ["chicken_is_heaven", ["View0"], {}, np.array([1,1,1,-100,-100,0,1,1,-100])]]]
#         cls.multiviewResults = [["Mumbo", {"":""}, {}, np.array([0,0,1,-100,-100,0,1,1,-100])]]
#
#     def test_simple(cls):
#         res = ResultAnalysis.getExampleErrorsBiclass(cls.usedBenchmarkArgumentDictionary, cls.monoViewResults,
#                                                   cls.multiviewResults)
#         cls.assertIn("chicken_is_heaven-View0", res)
#         cls.assertIn("Mumbo", res)
#         np.testing.assert_array_equal(res["Mumbo"], np.array([1,0,1,-100,-100,1,1,1,-100]))
#         np.testing.assert_array_equal(res["chicken_is_heaven-View0"], np.array([0,1,1,-100,-100,1,1,1,-100]))
