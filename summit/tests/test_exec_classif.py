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

import h5py
import numpy as np

from summit.tests.utils import rm_tmp, tmp_path, test_dataset

from summit.multiview_platform import exec_classif


# class Test_execute(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         rm_tmp()
#         os.mkdir(tmp_path)
#
#     def test_exec_simple(self):
#         exec_classif.exec_classif(["--config_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_config_simple.yml")])
#
#     def test_exec_iter(self):
#         exec_classif.exec_classif(["--config_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_config_iter.yml")])
#
#     def test_exec_hps(self):
#         exec_classif.exec_classif(["--config_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_config_hps.yml")])
#
#     @classmethod
#     def tearDown(self):
#         rm_tmp()

class Test_gen_single_monoview_arg_dictionary(unittest.TestCase):

    def test_no_config(self):
        conf = exec_classif.gen_single_monoview_arg_dictionary("classifier_name1",
                                                               {}, "nb_class",
                                                               "view_index",
                                                               "view_name",
                                                               "hps_kwargs")
        self.assertEqual(conf, {"classifier_name1": {},
                                "view_name": "view_name",
                                "view_index": "view_index",
                                "classifier_name": "classifier_name1",
                                "nb_class": "nb_class",
                                "hps_kwargs": "hps_kwargs"})


class Test_initBenchmark(unittest.TestCase):

    def test_benchmark_wanted(self):
        benchmark_output = exec_classif.init_benchmark(
            cl_type=[
                "monoview",
                "multiview"],
            monoview_algos=["decision_tree"],
            multiview_algos=["weighted_linear_late_fusion"])
        self.assertEqual(benchmark_output,
                         {'monoview': ['decision_tree'],
                          'multiview': ['weighted_linear_late_fusion']})


class Test_Functs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_initKWARGSFunc_no_monoview(self):
        benchmark = {"monoview": {}, "multiview": {}}
        args = exec_classif.init_kwargs_func({}, benchmark)
        self.assertEqual(args, {"monoview": {}, "multiview": {}})

    def test_init_kwargs(self):
        kwargs = exec_classif.init_kwargs(
            {"decision_tree": ""}, ["decision_tree"])
        self.assertEqual(kwargs, {"decision_tree": ""})
        kwargs = exec_classif.init_kwargs({"weighted_linear_late_fusion": ""},
                                          ["weighted_linear_late_fusion"], framework="multiview")
        self.assertEqual(kwargs, {"weighted_linear_late_fusion": ""})
        kwargs = exec_classif.init_kwargs({}, ["decision_tree"],)
        self.assertEqual(kwargs, {"decision_tree": {}})
        self.assertRaises(
            AttributeError,
            exec_classif.init_kwargs,
            {},
            ["test"])

    def test_arange_metrics(self):
        metrics = exec_classif.arange_metrics(
            {"accuracy_score": {}}, "accuracy_score")
        self.assertEqual(metrics, {"accuracy_score*": {}})
        self.assertRaises(
            ValueError, exec_classif.arange_metrics, {
                "test1": {}}, "test")

    def test_banchmark_init(self):
        from sklearn.model_selection import StratifiedKFold
        folds = StratifiedKFold(n_splits=2)
        res, lab_names = exec_classif.benchmark_init(directory=tmp_path,
                                                     classification_indices=[
                                                         np.array([0, 1, 2, 3]), np.array([4])],
                                                     labels=test_dataset.get_labels(),
                                                     labels_dictionary={
                                                         "yes": 0, "no": 1},
                                                     k_folds=folds,
                                                     dataset_var=test_dataset)
        self.assertEqual(res, [])
        self.assertEqual(lab_names, [0, 1])


class Test_InitArgumentDictionaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.benchmark = {
            "monoview": ["fake_monoview_classifier"],
            "multiview": {}}
        cls.views_dictionnary = {'test_view_0': 0, 'test_view': 1}
        cls.nb_class = 2
        cls.monoview_classifier_name = "fake_monoview_classifier"
        cls.monoview_classifier_arg_name = "fake_arg"
        cls.monoview_classifier_arg_value = "fake_value_1"
        cls.multiview_classifier_name = "fake_multiview_classifier"
        cls.multiview_classifier_arg_name = "fake_arg_mv"
        cls.multiview_classifier_arg_value = "fake_value_2"
        cls.init_kwargs = {
            'monoview': {
                cls.monoview_classifier_name:
                    {cls.monoview_classifier_arg_name: cls.monoview_classifier_arg_value}
            },
            "multiview": {
                cls.multiview_classifier_name: {
                    cls.multiview_classifier_arg_name: cls.multiview_classifier_arg_value}
            }
        }

    def test_init_argument_dictionaries_monoview(self):
        arguments = exec_classif.init_argument_dictionaries(self.benchmark,
                                                            self.views_dictionnary,
                                                            self.nb_class,
                                                            self.init_kwargs,
                                                            "None", {})
        expected_output = [{
            self.monoview_classifier_name: {
                self.monoview_classifier_arg_name: self.monoview_classifier_arg_value},
            "view_name": "test_view_0",
            'hps_kwargs': {},
            "classifier_name": self.monoview_classifier_name,
            "nb_class": self.nb_class,
            "view_index": 0},
            {self.monoview_classifier_name: {
                self.monoview_classifier_arg_name: self.monoview_classifier_arg_value},
             "view_name": "test_view",
             'hps_kwargs': {},
             "classifier_name": self.monoview_classifier_name,
             "nb_class": self.nb_class,
             "view_index": 1},
        ]
        self.assertEqual(arguments["monoview"], expected_output)

    def test_init_argument_dictionaries_multiview(self):
        self.benchmark["multiview"] = ["fake_multiview_classifier"]
        self.benchmark["monoview"] = {}
        arguments = exec_classif.init_argument_dictionaries(self.benchmark,
                                                            self.views_dictionnary,
                                                            self.nb_class,
                                                            self.init_kwargs,
                                                            "None", {})
        expected_output = [{
            "classifier_name": self.multiview_classifier_name,
            "view_indices": [0, 1],
            "view_names": ["test_view_0", "test_view"],
            "nb_class": self.nb_class,
            'hps_kwargs': {},
            "labels_names": None,
            self.multiview_classifier_name: {
                self.multiview_classifier_arg_name:
                self.multiview_classifier_arg_value},
        }, ]
        self.assertEqual(arguments["multiview"][0], expected_output[0])

    def test_init_argument_dictionaries_multiview_complex(self):
        self.multiview_classifier_arg_value = {
            "fake_value_2": "plif", "plaf": "plouf"}
        self.init_kwargs = {
            'monoview': {
                self.monoview_classifier_name:
                    {
                        self.monoview_classifier_arg_name: self.monoview_classifier_arg_value}
            },
            "multiview": {
                self.multiview_classifier_name: {
                    self.multiview_classifier_arg_name: self.multiview_classifier_arg_value}
            }
        }
        self.benchmark["multiview"] = ["fake_multiview_classifier"]
        self.benchmark["monoview"] = {}
        arguments = exec_classif.init_argument_dictionaries(self.benchmark,
                                                            self.views_dictionnary,
                                                            self.nb_class,
                                                            self.init_kwargs,
                                                            "None", {})
        expected_output = [{
            "classifier_name": self.multiview_classifier_name,
            "view_indices": [0, 1],
            'hps_kwargs': {},
            "view_names": ["test_view_0", "test_view"],
            "nb_class": self.nb_class,
            "labels_names": None,
            self.multiview_classifier_name: {
                self.multiview_classifier_arg_name:
                self.multiview_classifier_arg_value},
        }]
        self.assertEqual(arguments["multiview"][0], expected_output[0])


def fakeBenchmarkExec(core_index=-1, a=7, args=1):
    return [core_index, a]


def fakeBenchmarkExec_mutlicore(nb_cores=-1, a=6, args=1):
    return [nb_cores, a]


def fakeBenchmarkExec_monocore(
        dataset_var=1, a=4, args=1, track_tracebacks=False, nb_cores=1):
    return [a]


def fakegetResults(results, stats_iter,
                   benchmark_arguments_dictionaries, metrics, directory,
                   sample_ids, labels, feat_ids, view_names):
    return 3


def fakeDelete(a, b, c):
    return 9


def fake_analyze(a, b, c, d, sample_ids=None, labels=None, feature_ids=None,
                 view_names=None):
    pass


class Test_execBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()

        os.mkdir(tmp_path)
        cls.Dataset = test_dataset
        cls.argument_dictionaries = [{"a": 4, "args": {}}]
        cls.args = {
            "Base": {"name": "chicken_is_heaven", "type": "type", "pathf": "pathF"},
            "Classification": {"hps_iter": 1}}

    def test_simple(cls):
        res = exec_classif.exec_benchmark(nb_cores=1,
                                          stats_iter=2,
                                          benchmark_arguments_dictionaries=cls.argument_dictionaries,
                                          directory="",
                                          metrics=[[[1, 2], [3, 4, 5]]],
                                          dataset_var=cls.Dataset,
                                          track_tracebacks=6,
                                          # exec_one_benchmark=fakeBenchmarkExec,
                                          # exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                          exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                          analyze=fakegetResults,
                                          delete=fakeDelete,
                                          analyze_iterations=fake_analyze)
        cls.assertEqual(res, 3)

    def test_multiclass_no_iter(cls):
        cls.argument_dictionaries = [{"a": 10, "args": cls.args},
                                     {"a": 4, "args": cls.args}]
        res = exec_classif.exec_benchmark(nb_cores=1,
                                          stats_iter=1,
                                          benchmark_arguments_dictionaries=cls.argument_dictionaries,
                                          directory="",
                                          metrics=[[[1, 2], [3, 4, 5]]],
                                          dataset_var=cls.Dataset,
                                          track_tracebacks=6,
                                          # exec_one_benchmark=fakeBenchmarkExec,
                                          # exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                          exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                          analyze=fakegetResults,
                                          delete=fakeDelete,
                                          analyze_iterations=fake_analyze)
        cls.assertEqual(res, 3)

    def test_multiclass_and_iter(cls):
        cls.argument_dictionaries = [{"a": 10, "args": cls.args},
                                     {"a": 4, "args": cls.args},
                                     {"a": 55, "args": cls.args},
                                     {"a": 24, "args": cls.args}]
        res = exec_classif.exec_benchmark(nb_cores=1,
                                          stats_iter=2,
                                          benchmark_arguments_dictionaries=cls.argument_dictionaries,
                                          directory="",
                                          metrics=[[[1, 2], [3, 4, 5]]],
                                          dataset_var=cls.Dataset,
                                          track_tracebacks=6,
                                          # exec_one_benchmark=fakeBenchmarkExec,
                                          # exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                          exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                          analyze=fakegetResults,
                                          delete=fakeDelete,
                                          analyze_iterations=fake_analyze)
        cls.assertEqual(res, 3)

    def test_no_iter_biclass_multicore(cls):
        res = exec_classif.exec_benchmark(nb_cores=1,
                                          stats_iter=1,
                                          benchmark_arguments_dictionaries=cls.argument_dictionaries,
                                          directory="",
                                          metrics=[[[1, 2], [3, 4, 5]]],
                                          dataset_var=cls.Dataset,
                                          track_tracebacks=6,
                                          # exec_one_benchmark=fakeBenchmarkExec,
                                          # exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                          exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                          analyze=fakegetResults,
                                          delete=fakeDelete,
                                          analyze_iterations=fake_analyze)
        cls.assertEqual(res, 3)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()


def fakeExecMono(directory, name, labels_names, classification_indices, k_folds,
                 coreIndex, type, pathF, random_state, labels,
                 hyper_param_search="try", metrics="try", n_iter=1, **arguments):
    return ["Mono", arguments]


def fakeExecMulti(directory, coreIndex, name, classification_indices, k_folds,
                  type, pathF, labels_dictionary,
                  random_state, labels, hyper_param_search="", metrics=None,
                  n_iter=1, **arguments):
    return ["Multi", arguments]


def fakeInitMulti(args, benchmark, views, views_indices, argument_dictionaries,
                  random_state, directory, resultsMonoview,
                  classification_indices):
    return {"monoview": [{"try": 0}, {"try2": 100}],
            "multiview": [{"try3": 5}, {"try4": 10}]}


class FakeKfold():
    def __init__(self):
        self.n_splits = 2
        pass

    def split(self, X, Y):
        return [([X[0], X[1]], [X[2], X[3]]), (([X[2], X[3]], [X[0], X[1]]))]


class Test_set_element(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dictionary = {"a":
                          {"b": {
                              "c": {
                                  "d": {
                                      "e": 1,
                                      "f": [1]
                                  }
                              }
                          }}}
        cls.elements = {"a.b.c.d.e": 1, "a.b.c.d.f": [1]}

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        simplified_dict = {}
        for path, value in self.elements.items():
            simplified_dict = exec_classif.set_element(
                simplified_dict, path, value)
        self.assertEqual(simplified_dict, self.dictionary)


class Test_get_path_dict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dictionary = {"a":
                          {"b": {
                              "c": {
                                  "d": {
                                      "e": 1,
                                      "f": [1]
                                  }
                              }
                          }}}

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        path_dict = exec_classif.get_path_dict(self.dictionary)
        self.assertEqual(path_dict, {"a.b.c.d.e": 1, "a.b.c.d.f": [1]})


if __name__ == '__main__':
    unittest.main()
