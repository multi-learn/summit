import os
import unittest
import yaml
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from multiview_platform.tests.utils import rm_tmp, tmp_path
from multiview_platform.mono_multi_view_classifiers.utils import base


class FakeClassifier():
    pass

class Test_ResultAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rs = np.random.RandomState(42)
        cls.classifier = FakeClassifier()
        cls.n_examples = 50
        cls.n_classes = 3
        cls.train_length = 24
        cls.train_indices = cls.rs.choice(np.arange(cls.n_examples),
                                          size=cls.train_length,
                                          replace=False)
        cls.test_indices = np.array([i for i in range(cls.n_examples)
                                     if i not in cls.train_indices])
        cls.test_length = cls.test_indices.shape[0]
        cls.classification_indices = [cls.train_indices, cls.test_indices]
        cls.n_splits = 5
        cls.k_folds = StratifiedKFold(n_splits=cls.n_splits, )
        cls.hps_method = "randomized_search"
        cls.metrics_list = [("accuracy_score", {}), ("f1_score", {})]
        cls.n_iter = 6
        cls.class_label_names = ["class{}".format(ind+1)
                                  for ind in range(cls.n_classes)]
        cls.train_pred = np.random.randint(0, cls.n_classes,
                                           size=cls.train_length)
        cls.test_pred = np.random.randint(0, cls.n_classes,
                                          size=cls.test_length)
        cls.directory = "fake_directory"
        cls.labels = np.random.randint(0, cls.n_classes,
                                           size=cls.n_examples)
        cls.database_name = "test_database"
        cls.nb_cores = 0.5
        cls.duration = -4
        cls.train_accuracy = accuracy_score(cls.labels[cls.train_indices],
                                            cls.train_pred)
        cls.test_accuracy = accuracy_score(cls.labels[cls.test_indices],
                                            cls.test_pred)
        cls.train_f1 = f1_score(cls.labels[cls.train_indices],
                                            cls.train_pred, average='micro')
        cls.test_f1 = f1_score(cls.labels[cls.test_indices],
                                           cls.test_pred, average='micro')

    def test_simple(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method, self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.train_pred, self.test_pred, self.directory,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration)

    def test_get_metric_scores(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method,
                                 self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.train_pred, self.test_pred,
                                 self.directory,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration)
        train_score, test_score = RA.get_metric_scores("accuracy_score", {})
        self.assertEqual(train_score, self.train_accuracy)
        self.assertEqual(test_score, self.test_accuracy)

    def test_get_all_metrics_scores(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method,
                                 self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.train_pred, self.test_pred,
                                 self.directory,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration)
        RA.get_all_metrics_scores()
        self.assertEqual(RA.metric_scores["accuracy_score"][0],
                         self.train_accuracy)
        self.assertEqual(RA.metric_scores["accuracy_score"][1],
                         self.test_accuracy)
        self.assertEqual(RA.metric_scores["f1_score"][0],
                         self.train_f1)
        self.assertEqual(RA.metric_scores["f1_score"][1],
                         self.test_f1)


class Test_BaseClassifier(unittest.TestCase):

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