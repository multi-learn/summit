import os
import unittest
import yaml
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from multiview_platform.tests.utils import rm_tmp, tmp_path
from multiview_platform.mono_multi_view_classifiers.utils import base


class Test_ResultAnalyzer(unittest.TestCase):
    pass

class Test_BaseEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_estimator = "DecisionTreeClassifier"
        cls.base_estimator_config = {"max_depth":10,
                                     "splitter": "best"}
        cls.est = base.BaseClassifier()

    def test_simple(self):
        base_estim = self.est.get_base_estimator(self.base_estimator,
                                            self.base_estimator_config)
        self.assertTrue(isinstance(base_estim, DecisionTreeClassifier))
        self.assertEqual(base_estim.max_depth, 10)
        self.assertEqual(base_estim.splitter, "best")

    def test_class(self):
        base_estimator = DecisionTreeClassifier(max_depth=15, splitter="random")
        base_estim = self.est.get_base_estimator(base_estimator,
                                            self.base_estimator_config)
        self.assertTrue(isinstance(base_estim, DecisionTreeClassifier))
        self.assertEqual(base_estim.max_depth, 10)
        self.assertEqual(base_estim.splitter, "best")

    def test_wrong_args(self):
        base_estimator_config = {"n_estimators": 10,
                                 "splitter": "best"}
        with self.assertRaises(TypeError):
            base_estim = self.est.get_base_estimator(self.base_estimator,
                                                     base_estimator_config)