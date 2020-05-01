import os
import unittest

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

from summit.tests.utils import rm_tmp, tmp_path, test_dataset

from summit.multiview_platform.multiview import multiview_utils


class FakeMVClassif(multiview_utils.BaseMultiviewClassifier):

    def __init__(self, mc=True):
        self.mc = mc
        pass

    def fit(self, X, y):
        if not self.mc:
            raise ValueError
        else:
            pass


class TestBaseMultiviewClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_accepts_multiclass(self):
        rs = np.random.RandomState(42)
        accepts = FakeMVClassif().accepts_multi_class(rs)
        self.assertEqual(accepts, True)
        accepts = FakeMVClassif(mc=False).accepts_multi_class(rs)
        self.assertEqual(accepts, False)
        self.assertRaises(ValueError, FakeMVClassif(
            mc=False).accepts_multi_class, rs, **{"n_samples": 2, "n_classes": 3})


class TestConfigGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rs = np.random.RandomState(42)

    def test_simple(self):
        cfg_gen = multiview_utils.ConfigGenerator(
            ["decision_tree", "decision_tree"])
        sample = cfg_gen.rvs(self.rs)
        self.assertEqual(sample, {'decision_tree': {'criterion': 'entropy',
                                                    'max_depth': 103,
                                                    'splitter': 'best'}})


class TestFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.rs = np.random.RandomState(42)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_get_available_monoview_classifiers(self):
        avail = multiview_utils.get_available_monoview_classifiers()
        self.assertEqual(avail, ['adaboost',
                                 'decision_tree',
                                 'gradient_boosting',
                                 'knn',
                                 'lasso',
                                 'random_forest',
                                 'sgd',
                                 'svm_linear',
                                 'svm_poly',
                                 'svm_rbf'])
        avail = multiview_utils.get_available_monoview_classifiers(
            need_probas=True)
        self.assertEqual(avail, ['adaboost',
                                 'decision_tree',
                                 'gradient_boosting',
                                 'knn',
                                 'random_forest',
                                 'svm_linear',
                                 'svm_poly',
                                 'svm_rbf'])
