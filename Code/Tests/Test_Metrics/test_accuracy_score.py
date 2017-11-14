import unittest
import os

from ...MonoMultiViewClassifiers.Metrics import accuracy_score

# class Test_accuracy_score(unittest.TestCase):
#
#     def setUpClass(cls):
#         pass
#
#
#     def test_all_scores(cls):
#         for filename in os.listdir("Code/MonoMultiviewClassifiers/Metrics/"):
#             if filename != "__init__.py":
#                 metric_name = filename[:-3]
#                 metric_module = getattr(Metrics, metric_name)
#                 cls.score_test(metric_module)
#
#
#     def score_test(cls, metric_module):
#         pass
