# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#
import os
import unittest
import yaml
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from summit.tests.utils import rm_tmp, tmp_path
from summit.multiview_platform.utils import base


class FakeClassifier(base.BaseClassifier):
    def __init__(self, no_params=False, accepts_mc=True):
        if no_params:
            self.param_names = []
            self.classed_params = []
        else:
            self.param_names = ["test1", "test2"]
            self.classed_params = ["test2"]
            self.weird_strings = []
        self.accepts_mc = accepts_mc

    def get_params(self, deep=True):
        return {"test1": 10,
                "test2": "test"}

    def fit(self, X, y):
        if np.unique(y).shape[0] > 2 and not self.accepts_mc:
            raise ValueError('Does not accept MC')
        else:
            return self


class FakeDetector:
    def __init__(self):
        self.best_params_ = {"test1": 10,
                             "test2": "test"}
        self.cv_results_ = {"param_test1": [10],
                            "param_test2": ["test"]}


class FakeResultAnalyzer(base.ResultAnalyser):

    def get_view_specific_info(self):
        return "test"

    def get_base_string(self):
        return 'test2'


class Test_ResultAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rs = np.random.RandomState(42)
        cls.classifier = FakeClassifier()
        cls.n_samples = 50
        cls.n_classes = 3
        cls.train_length = 24
        cls.train_indices = cls.rs.choice(np.arange(cls.n_samples),
                                          size=cls.train_length,
                                          replace=False)
        cls.test_indices = np.array([i for i in range(cls.n_samples)
                                     if i not in cls.train_indices])
        cls.test_length = cls.test_indices.shape[0]
        cls.classification_indices = [cls.train_indices, cls.test_indices]
        cls.n_splits = 5
        cls.k_folds = StratifiedKFold(n_splits=cls.n_splits, )
        cls.hps_method = "randomized_search"
        cls.metrics_list = {"accuracy_score": {}, "f1_score*": {}}
        cls.n_iter = 6
        cls.class_label_names = ["class{}".format(ind + 1)
                                 for ind in range(cls.n_classes)]
        cls.pred = cls.rs.randint(0, cls.n_classes,
                                  size=cls.n_samples)
        cls.directory = "fake_directory"
        cls.base_file_name = "fake_file"
        cls.labels = cls.rs.randint(0, cls.n_classes,
                                    size=cls.n_samples)
        cls.database_name = "test_database"
        cls.nb_cores = 0.5
        cls.duration = -4
        cls.train_accuracy = accuracy_score(cls.labels[cls.train_indices],
                                            cls.pred[cls.train_indices])
        cls.test_accuracy = accuracy_score(cls.labels[cls.test_indices],
                                           cls.pred[cls.test_indices])
        cls.train_f1 = f1_score(cls.labels[cls.train_indices],
                                cls.pred[cls.train_indices], average='micro')
        cls.test_f1 = f1_score(cls.labels[cls.test_indices],
                               cls.pred[cls.test_indices], average='micro')

    def test_simple(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method, self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.pred, self.directory,
                                 self.base_file_name, self.labels,
                                 self.database_name, self.nb_cores,
                                 self.duration, [""])

    def test_get_metric_scores(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method,
                                 self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.pred,
                                 self.directory, self.base_file_name,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration, [""])
        cl_train, cl_test, train_score, test_score = RA.get_metric_score(
            "accuracy_score", {})
        np.testing.assert_array_equal(train_score, self.train_accuracy)
        np.testing.assert_array_equal(test_score, self.test_accuracy)

    def test_get_all_metrics_scores(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method,
                                 self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.pred,
                                 self.directory, self.base_file_name,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration, [""])
        RA.get_all_metrics_scores()
        self.assertEqual(RA.metric_scores["accuracy_score"][0],
                         self.train_accuracy)
        self.assertEqual(RA.metric_scores["accuracy_score"][1],
                         self.test_accuracy)
        self.assertEqual(RA.metric_scores["f1_score*"][0],
                         self.train_f1)
        self.assertEqual(RA.metric_scores["f1_score*"][1],
                         self.test_f1)

    def test_print_metrics_scores(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method,
                                 self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.pred,
                                 self.directory, self.base_file_name,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration, [''])
        RA.get_all_metrics_scores()
        string = RA.print_metric_score()
        self.assertEqual(string, '\n\n\tFor Accuracy score using {}, (higher is better) : \n\t\t- Score on train : 0.25\n\t\t- Score on test : 0.2692307692307692\n\n\tFor F1 score using average: micro, {} (higher is better) : \n\t\t- Score on train : 0.25\n\t\t- Score on test : 0.2692307692307692\n\nTest set confusion matrix : \n\n╒════════╤══════════╤══════════╤══════════╕\n│        │   class1 │   class2 │   class3 │\n╞════════╪══════════╪══════════╪══════════╡\n│ class1 │        3 │        1 │        2 │\n├────────┼──────────┼──────────┼──────────┤\n│ class2 │        3 │        2 │        2 │\n├────────┼──────────┼──────────┼──────────┤\n│ class3 │        3 │        8 │        2 │\n╘════════╧══════════╧══════════╧══════════╛\n\n')

    def test_get_db_config_string(self):
        RA = FakeResultAnalyzer(self.classifier, self.classification_indices,
                                self.k_folds, self.hps_method,
                                self.metrics_list,
                                self.n_iter, self.class_label_names,
                                self.pred,
                                self.directory, self.base_file_name,
                                self.labels, self.database_name,
                                self.nb_cores, self.duration, [''])
        self.assertEqual(
            RA.get_db_config_string(),
            'Database configuration : \n\t- Database name : test_database\ntest\t- Learning Rate : 0.48\n\t- Labels used : class1, class2, class3\n\t- Number of cross validation folds : 5\n\n')

    def test_get_classifier_config_string(self):
        RA = base.ResultAnalyser(self.classifier, self.classification_indices,
                                 self.k_folds, self.hps_method,
                                 self.metrics_list,
                                 self.n_iter, self.class_label_names,
                                 self.pred,
                                 self.directory, self.base_file_name,
                                 self.labels, self.database_name,
                                 self.nb_cores, self.duration, [''])
        self.assertEqual(
            RA.get_classifier_config_string(),
            'Classifier configuration : \n\t- FakeClassifier with test1 : 10, test2 : test\n\t- Executed on 0.5 core(s) \n\t- Got configuration using randomized search with 6  iterations \n')

    def test_analyze(self):
        RA = FakeResultAnalyzer(self.classifier, self.classification_indices,
                                self.k_folds, self.hps_method,
                                self.metrics_list,
                                self.n_iter, self.class_label_names,
                                self.pred,
                                self.directory, self.base_file_name,
                                self.labels, self.database_name,
                                self.nb_cores, self.duration, [""])
        str_analysis, img_analysis, metric_scores, class_metric_scores, conf_mat = RA.analyze()
        self.assertEqual(str_analysis, 'test2Database configuration : \n\t- Database name : test_database\ntest\t- Learning Rate : 0.48\n\t- Labels used : class1, class2, class3\n\t- Number of cross validation folds : 5\n\nClassifier configuration : \n\t- FakeClassifier with test1 : 10, test2 : test\n\t- Executed on 0.5 core(s) \n\t- Got configuration using randomized search with 6  iterations \n\n\n\tFor Accuracy score using {}, (higher is better) : \n\t\t- Score on train : 0.25\n\t\t- Score on test : 0.2692307692307692\n\n\tFor F1 score using average: micro, {} (higher is better) : \n\t\t- Score on train : 0.25\n\t\t- Score on test : 0.2692307692307692\n\nTest set confusion matrix : \n\n╒════════╤══════════╤══════════╤══════════╕\n│        │   class1 │   class2 │   class3 │\n╞════════╪══════════╪══════════╪══════════╡\n│ class1 │        3 │        1 │        2 │\n├────────┼──────────┼──────────┼──────────┤\n│ class2 │        3 │        2 │        2 │\n├────────┼──────────┼──────────┼──────────┤\n│ class3 │        3 │        8 │        2 │\n╘════════╧══════════╧══════════╧══════════╛\n\n\n\n Classification took -1 day, 23:59:56\n\n Classifier Interpretation : \n')


class Test_BaseClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.estimator = "DecisionTreeClassifier"
        cls.base_estimator_config = {"max_depth": 10,
                                     "splitter": "best"}
        cls.est = base.BaseClassifier()
        cls.rs = np.random.RandomState(42)

    def test_simple(self):
        base_estim = self.est.get_base_estimator(self.estimator,
                                                 self.base_estimator_config)
        self.assertTrue(isinstance(base_estim, DecisionTreeClassifier))
        self.assertEqual(base_estim.max_depth, 10)
        self.assertEqual(base_estim.splitter, "best")

    def test_gen_best_params(self):
        fake_class = FakeClassifier()
        best_params = fake_class.gen_best_params(FakeDetector())
        self.assertEqual(best_params, {"test1": 10, "test2": "test"})

    def test_gen_params_from_detector(self):
        fake_class = FakeClassifier()
        params = fake_class.gen_params_from_detector(FakeDetector())
        self.assertEqual(params, [("test1", np.array([10])),
                                  ("test2", np.array(["str"], dtype='<U3'))])
        params = FakeClassifier(
            no_params=True).gen_params_from_detector(
            FakeDetector())
        self.assertEqual(params, [()])

    def test_params_to_string(self):
        fake_class = FakeClassifier()
        string = fake_class.params_to_string()
        self.assertEqual(string, "test1 : 10, test2 : test")

    def test_get_iterpret(self):
        fake_class = FakeClassifier()
        self.assertEqual("", fake_class.get_interpretation("", "", "",""))

    def test_accepts_mutliclass(self):
        accepts = FakeClassifier().accepts_multi_class(self.rs)
        self.assertEqual(accepts, True)
        accepts = FakeClassifier(accepts_mc=False).accepts_multi_class(self.rs)
        self.assertEqual(accepts, False)
        self.assertRaises(ValueError,
                          FakeClassifier().accepts_multi_class,
                          self.rs,
                          **{"n_samples": 2})

    def test_class(self):
        estimator = DecisionTreeClassifier(
            max_depth=15, splitter="random")
        base_estim = self.est.get_base_estimator(estimator,
                                                 self.base_estimator_config)
        self.assertTrue(isinstance(base_estim, DecisionTreeClassifier))
        self.assertEqual(base_estim.max_depth, 10)
        self.assertEqual(base_estim.splitter, "best")

    def test_wrong_args(self):
        base_estimator_config = {"n_estimators": 10,
                                 "splitter": "best"}
        with self.assertRaises(TypeError):
            base_estim = self.est.get_base_estimator(self.estimator,
                                                     base_estimator_config)

    def test_get_config(self):
        conf = FakeClassifier(no_params=True).get_config()
        self.assertEqual(conf, 'FakeClassifier with no config.')


class Test_Functions(unittest.TestCase):

    def test_get_name(self):
        classed_list = ["test", 42]
        np.testing.assert_array_equal(base.get_names(classed_list),
                                      np.array(["str", "int"], dtype="<U3"))

    def test_get_metric(self):
        from summit.multiview_platform.metrics import accuracy_score
        metrics_dict = {"accuracy_score*": {}}
        self.assertEqual(base.get_metric(metrics_dict), (accuracy_score, {}))
