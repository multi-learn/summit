#
# import numpy as np
# import unittest
#
# from summit.multiview_platform.multiview.additions import \
#     diversity_utils
#
#
# def fake_measure(a, b, c, d, e):
#     return 42
#
#
# class Test_global_div_measure(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         cls.allClassifiersNames = [["SCM", "SVM", "DT"], ["SCM", "SVM", "DT"]]
#         cls.views_indices = np.array([0, 1])
#         cls.classifiersDecisions = np.zeros(
#             (cls.views_indices.shape[0], len(cls.allClassifiersNames), 3, 6),
#             dtype=int)
#         for classifer_index, classifier in enumerate(cls.allClassifiersNames):
#             for view_index, view in enumerate(cls.views_indices):
#                 cls.classifiersDecisions[
#                     view_index, classifer_index] = np.array([
#                     cls.random_state.randint(0, 2, 6),
#                     cls.random_state.randint(0, 2, 6),
#                     cls.random_state.randint(0, 2, 6)])
#         cls.folds_ground_truth = np.array(
#             [np.array([1, 1, 1, 0, 0, 0]) for _ in range(3)])
#         cls.classification_indices = np.array([])
#         cls.measurement = fake_measure
#
#     def test_simple(cls):
#         clf_names, diversity_measure = diversity_utils.global_div_measure(
#             cls.allClassifiersNames,
#             cls.classifiersDecisions,
#             cls.measurement,
#             cls.folds_ground_truth)
#         cls.assertEqual(len(clf_names), 2)
#         cls.assertEqual(diversity_measure, 42)
