import os
import unittest

import h5py
import numpy as np

from .utils import rm_tmp, tmp_path, test_dataset

from multiview_platform.mono_multi_view_classifiers import exec_classif


class Test_initBenchmark(unittest.TestCase):

    def test_benchmark_wanted(self):
        # benchmark_output = ExecClassif.init_benchmark(self.args)
        self.assertEqual(1, 1)


class Test_initKWARGS(unittest.TestCase):

    def test_initKWARGSFunc_no_monoview(self):
        benchmark = {"monoview": {}, "multiview": {}}
        args = exec_classif.init_kwargs_func({}, benchmark)
        self.assertEqual(args, {"monoview": {}, "multiview": {}})


class Test_InitArgumentDictionaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.benchmark = {"monoview": ["fake_monoview_classifier"], "multiview": {}}
        cls.views_dictionnary = {'test_view_0': 0, 'test_view': 1}
        cls.nb_class = 2
        cls.monoview_classifier_name = "fake_monoview_classifier"
        cls.monoview_classifier_arg_name = "fake_arg"
        cls.monoview_classifier_arg_value = ["fake_value_1"]
        cls.multiview_classifier_name = "fake_multiview_classifier"
        cls.multiview_classifier_arg_name = "fake_arg_mv"
        cls.multiview_classifier_arg_value = ["fake_value_2"]
        cls.init_kwargs = {
            'monoview':{
                cls.monoview_classifier_name:
                    {cls.monoview_classifier_arg_name:cls.monoview_classifier_arg_value}
            },
            "multiview":{
                cls.multiview_classifier_name:{
                    cls.multiview_classifier_arg_name:cls.multiview_classifier_arg_value}
            }
        }

    def test_init_argument_dictionaries_monoview(self):
        arguments = exec_classif.init_argument_dictionaries(self.benchmark,
                                                            self.views_dictionnary,
                                                            self.nb_class,
                                                            self.init_kwargs)
        expected_output = [{
                self.monoview_classifier_name: {
                    self.monoview_classifier_arg_name:self.monoview_classifier_arg_value[0]},
                "view_name": "test_view_0",
                "classifier_name": self.monoview_classifier_name,
                "nb_class": self.nb_class,
                "view_index": 0},
                {self.monoview_classifier_name: {
                    self.monoview_classifier_arg_name: self.monoview_classifier_arg_value[0]},
                "view_name": "test_view",
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
                                                            self.init_kwargs)
        expected_output = [{
                "classifier_name": self.multiview_classifier_name,
                "view_indices": [0,1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names":None,
                self.multiview_classifier_name: {
                    self.multiview_classifier_arg_name:
                        self.multiview_classifier_arg_value[0]},
        },]
        self.assertEqual(arguments["multiview"][0], expected_output[0])

    def test_init_argument_dictionaries_multiview_multiple(self):
        self.multiview_classifier_arg_value = ["fake_value_2", "fake_arg_value_3"]
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
                                                            self.init_kwargs)
        expected_output = [{
                "classifier_name": self.multiview_classifier_name+"_fake_value_2",
                "view_indices": [0,1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names":None,
                self.multiview_classifier_name + "_fake_value_2": {
                    self.multiview_classifier_arg_name:
                        self.multiview_classifier_arg_value[0]},
        },
            {
                "classifier_name": self.multiview_classifier_name+"_fake_arg_value_3",
                "view_indices": [0, 1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names": None,
                self.multiview_classifier_name+"_fake_arg_value_3": {
                    self.multiview_classifier_arg_name:
                        self.multiview_classifier_arg_value[1]},
            }
        ]
        self.assertEqual(arguments["multiview"][0], expected_output[0])

    def test_init_argument_dictionaries_multiview_complex(self):
        self.multiview_classifier_arg_value = {"fake_value_2":"plif", "plaf":"plouf"}
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
                                                            self.init_kwargs)
        expected_output = [{
                "classifier_name": self.multiview_classifier_name,
                "view_indices": [0,1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names":None,
                self.multiview_classifier_name: {
                    self.multiview_classifier_arg_name:
                        self.multiview_classifier_arg_value},
        }]
        self.assertEqual(arguments["multiview"][0], expected_output[0])

    def test_init_argument_dictionaries_multiview_multiple_complex(self):
        self.multiview_classifier_arg_value = {"fake_value_2":["plif", "pluf"], "plaf":"plouf"}
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
                                                            self.init_kwargs)
        expected_output = [{
                "classifier_name": self.multiview_classifier_name+"_plif_plouf",
                "view_indices": [0,1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names":None,
                self.multiview_classifier_name + "_plif_plouf": {
                    self.multiview_classifier_arg_name:
                        {"fake_value_2": "plif", "plaf": "plouf"}},
        },
            {
                "classifier_name": self.multiview_classifier_name+"_pluf_plouf",
                "view_indices": [0, 1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names": None,
                self.multiview_classifier_name+"_pluf_plouf": {
                    self.multiview_classifier_arg_name:
                        {"fake_value_2":"pluf", "plaf":"plouf"}},
            }
        ]
        self.assertEqual(arguments["multiview"][0], expected_output[0])


def fakeBenchmarkExec(core_index=-1, a=7, args=1):
    return [core_index, a]


def fakeBenchmarkExec_mutlicore(nb_cores=-1, a=6, args=1):
    return [nb_cores, a]


def fakeBenchmarkExec_monocore(dataset_var=1, a=4, args=1):
    return [a]


def fakegetResults(results, stats_iter, nb_multiclass,
                   benchmark_arguments_dictionaries, multi_class_labels, metrics,
                   classification_indices, directories, directory,
                   labels_dictionary, nb_examples, nb_labels):
    return 3


def fakeDelete(a, b, c):
    return 9


class Test_execBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()

        os.mkdir(tmp_path)
        cls.Dataset = test_dataset
        cls.argument_dictionaries = [{"a": 4, "args": {}}]
        cls.args = {
            "Base":{"name": "chicken_is_heaven", "type": "type", "pathf": "pathF"},
            "Classification":{"hps_iter": 1}}

    def test_simple(cls):
        res = exec_classif.exec_benchmark(1, 2, 3, cls.argument_dictionaries,
                                         [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                         10, cls.Dataset,
                                         exec_one_benchmark=fakeBenchmarkExec,
                                         exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                         exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                         get_results=fakegetResults,
                                         delete=fakeDelete)
        cls.assertEqual(res, 3)

    def test_multiclass_no_iter(cls):
        cls.argument_dictionaries = [{"a": 10, "args": cls.args},
                                    {"a": 4, "args": cls.args}]
        res = exec_classif.exec_benchmark(2, 1, 2, cls.argument_dictionaries,
                                         [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                         10, cls.Dataset,
                                         exec_one_benchmark=fakeBenchmarkExec,
                                         exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                         exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                         get_results=fakegetResults,
                                         delete=fakeDelete)
        cls.assertEqual(res, 3)

    def test_multiclass_and_iter(cls):
        cls.argument_dictionaries = [{"a": 10, "args": cls.args},
                                    {"a": 4, "args": cls.args},
                                    {"a": 55, "args": cls.args},
                                    {"a": 24, "args": cls.args}]
        res = exec_classif.exec_benchmark(2, 2, 2, cls.argument_dictionaries,
                                         [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                         10, cls.Dataset,
                                         exec_one_benchmark=fakeBenchmarkExec,
                                         exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                         exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                         get_results=fakegetResults,
                                         delete=fakeDelete)
        cls.assertEqual(res, 3)

    def test_no_iter_biclass_multicore(cls):
        res = exec_classif.exec_benchmark(2, 1, 1, cls.argument_dictionaries,
                                         [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                         10, cls.Dataset,
                                         exec_one_benchmark=fakeBenchmarkExec,
                                         exec_one_benchmark_multicore=fakeBenchmarkExec_mutlicore,
                                         exec_one_benchmark_mono_core=fakeBenchmarkExec_monocore,
                                         get_results=fakegetResults,
                                         delete=fakeDelete)
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


class Test_execOneBenchmark(unittest.TestCase):

    @classmethod
    def setUp(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.args = {
            "Base": {"name": "chicken_is_heaven", "type": "type",
                     "pathf": "pathF"},
            "Classification": {"hps_iter": 1}}

    def test_simple(cls):
        flag, results = exec_classif.exec_one_benchmark(core_index=10,
                                                      labels_dictionary={
                                                                   0: "a",
                                                                   1: "b"},
                                                      directory=tmp_path,
                                                      classification_indices=(
                                                               [1, 2, 3, 4],
                                                               [0, 5, 6, 7, 8]),
                                                               args=cls.args,
                                                               k_folds=FakeKfold(),
                                                               random_state="try",
                                                               hyper_param_search="try",
                                                               metrics="try",
                                                               argument_dictionaries={
                                                                   "Monoview": [
                                                                       {
                                                                           "try": 0},
                                                                       {
                                                                           "try2": 100}],
                                                                   "multiview":[{
                                                                           "try3": 5},
                                                                       {
                                                                           "try4": 10}]},
                                                      benchmark="try",
                                                      views="try",
                                                      views_indices="try",
                                                      flag=None,
                                                      labels=np.array(
                                                                   [0, 1, 2, 1,
                                                                    2, 2, 2, 12,
                                                                    1, 2, 1, 1,
                                                                    2, 1, 21]),
                                                      exec_monoview_multicore=fakeExecMono,
                                                      exec_multiview_multicore=fakeExecMulti,
                                                      init_multiview_arguments=fakeInitMulti)

        cls.assertEqual(flag, None)
        cls.assertEqual(results ,
                        [['Mono', {'try': 0}], ['Mono', {'try2': 100}],
                         ['Multi', {'try3': 5}], ['Multi', {'try4': 10}]])

    @classmethod
    def tearDown(cls):
        path = tmp_path
        for file_name in os.listdir(path):
            dir_path = os.path.join(path, file_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file_name))
                os.rmdir(dir_path)
            else:
                os.remove(os.path.join(path, file_name))
        os.rmdir(path)


class Test_execOneBenchmark_multicore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.args = {
            "Base": {"name": "chicken_is_heaven", "type": "type",
                     "pathf": "pathF"},
            "Classification": {"hps_iter": 1}}

    def test_simple(cls):
        flag, results = exec_classif.exec_one_benchmark_multicore(
            nb_cores=2,
            labels_dictionary={0: "a", 1: "b"},
            directory=tmp_path,
            classification_indices=([1, 2, 3, 4], [0, 10, 20, 30, 40]),
            args=cls.args,
            k_folds=FakeKfold(),
            random_state="try",
            hyper_param_search="try",
            metrics="try",
            argument_dictionaries={
                                                                   "monoview": [
                                                                       {
                                                                           "try": 0},
                                                                       {
                                                                           "try2": 100}],
                                                                   "multiview":[{
                                                                           "try3": 5},
                                                                       {
                                                                           "try4": 10}]},
            benchmark="try",
            views="try",
            views_indices="try",
            flag=None,
            labels=np.array([0, 1, 2, 3, 4, 2, 2, 12, 1, 2, 1, 1, 2, 1, 21]),
            exec_monoview_multicore=fakeExecMono,
            exec_multiview_multicore=fakeExecMulti,
            init_multiview_arguments=fakeInitMulti)

        cls.assertEqual(flag, None)
        cls.assertEqual(results ,
                        [['Mono', {'try': 0}], ['Mono', {'try2': 100}],
                         ['Multi', {'try3': 5}], ['Multi', {'try4': 10}]])

    @classmethod
    def tearDown(cls):
        path = tmp_path
        for file_name in os.listdir(path):
            dir_path = os.path.join(path, file_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file_name))
                os.rmdir(dir_path)
            else:
                os.remove(os.path.join(path, file_name))
        os.rmdir(path)


class Test_set_element(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dictionary = {"a":
                              {"b":{
                                  "c":{
                                      "d":{
                                          "e":1,
                                          "f":[1]
                                      }
                                  }
                              }}}
        cls.elements = {"a.b.c.d.e":1, "a.b.c.d.f":[1]}

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        simplified_dict = {}
        for path, value in self.elements.items():
            simplified_dict = exec_classif.set_element(simplified_dict, path, value)
        self.assertEqual(simplified_dict, self.dictionary)


class Test_get_path_dict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dictionary = {"a":
                              {"b":{
                                  "c":{
                                      "d":{
                                          "e":1,
                                          "f":[1]
                                      }
                                  }
                              }}}

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        path_dict = exec_classif.get_path_dict(self.dictionary)
        self.assertEqual(path_dict, {"a.b.c.d.e":1, "a.b.c.d.f":[1]})


#
# class Test_analyzeMulticlass(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.flags = [[0, [0,1]], [0, [0,2]], [0, [0,3]], [0, [1,2]], [0, [1,3]], [0, [2,3]],
#                      [1, [0,1]], [1, [0,2]], [1, [0,3]], [1, [1,2]], [1, [1,3]], [1, [2,3]]]
#         cls.preds = [np.array([1, 0, 1, 1, 1]), np.array([1,0,0,1,1]), np.array([1,0,0,0,1]), np.array([1,1,0,1,1]),
#                      np.array([1,1,0,0,1]), np.array([1,1,1,0,1])] + \
#                     [np.array([0 in range(5)]) for i in range(6)]
#         cls.preds2 = [np.array([0 in range(5)]) for i in range(6)] + \
#                     [np.array([1, 0, 1, 1, 1]), np.array([1,0,0,1,1]),
#                      np.array([1,0,0,0,1]), np.array([1,1,0,1,1]), np.array([1,1,0,0,1]), np.array([1,1,1,0,1])]
#         cls.classifiers_names = ["chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven",
#                                 "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven",
#                                 "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven",]
#         cls.classifiersNames2 = ["cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease",
#                                 "cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease",
#                                 "cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease",
#                                 "cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease"]
#         cls.results = [[flag, [["", [name, "", "", pred]], ["", [name1, "", "", pred1]]], ["", ""]]
#                        for flag, name, pred, name1, pred1 in zip(cls.flags, cls.classifiers_names, cls.preds,
#                                                                  cls.classifiersNames2, cls.preds2)]
#         # cls.results = [[flag, ["", ["", name, "", pred]], ""] for flag, pred, name in
#         #                zip(cls.flags, cls.preds, cls.classifiers_names)]
#         cls.statsIter = 2
#         cls.nbExample = 5
#         cls.nbLabels = 4
#         cls.true_labels = np.array([0,1,2,3,0])
#         cls.metrics = [["accuracy_score"]]
#
#     def test_simple(cls):
#         multiclassResults = ExecClassif.analyzeMulticlass(cls.results, cls.statsIter, cls.nbExample, cls.nbLabels, cls.true_labels, [["accuracy_score"]])
#         np.testing.assert_array_equal(multiclassResults[1]["chicken_is_heaven"]["labels"], cls.true_labels)
#
# class Test_genMetricsScores(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.multiclass_labels = np.array([0,1,2,3,4,5,2,1,3])
#         cls.wrong_labels = np.array([1,3,3,4,5,0,2,4,3])
#         cls.multiclassResults = [{"chicken_is_heaven":
#                                       {"labels": cls.multiclass_labels}}]
#         cls.true_labels = np.array([0,2,2,3,4,5,1,3,2])
#         cls.metrics = [["accuracy_score"]]
#         cls.score_to_get = accuracy_score(cls.true_labels, cls.multiclass_labels)
#
#     def test_simple(cls):
#         multiclassResults = ExecClassif.genMetricsScores(cls.multiclassResults, cls.true_labels, cls.metrics)
#         cls.assertEqual(cls.score_to_get, multiclassResults[0]["chicken_is_heaven"]["metricsScores"]["accuracy_score"])
#
#     def test_multiple_clf(cls):
#         cls.multiclassResults = [{"chicken_is_heaven": {"labels": cls.multiclass_labels},
#                                   "cheese_is_no_disease": {"labels": cls.wrong_labels}},
#                                  ]
#         multiclassResults = ExecClassif.genMetricsScores(cls.multiclassResults, cls.true_labels, cls.metrics)
#         cls.assertEqual(0, multiclassResults[0]["cheese_is_no_disease"]["metricsScores"]["accuracy_score"])
#         cls.assertEqual(cls.score_to_get, multiclassResults[0]["chicken_is_heaven"]["metricsScores"]["accuracy_score"])
#
#     def test_multiple_metrics(cls):
#         from sklearn.metrics import f1_score
#         cls.score_to_get_f1 = f1_score(cls.true_labels, cls.multiclass_labels, average="micro")
#         cls.metrics = [["accuracy_score"], ["f1_score"]]
#         multiclassResults = ExecClassif.genMetricsScores(cls.multiclassResults, cls.true_labels, cls.metrics)
#         cls.assertEqual(cls.score_to_get, multiclassResults[0]["chicken_is_heaven"]["metricsScores"]["accuracy_score"])
#         cls.assertEqual(cls.score_to_get_f1, multiclassResults[0]["chicken_is_heaven"]["metricsScores"]["f1_score"])
#
#     def test_multiple_iterations(cls):
#         cls.multiclassResults = [{"chicken_is_heaven": {"labels": cls.multiclass_labels}},
#                                  {"chicken_is_heaven": {"labels": cls.wrong_labels}},
#                                  ]
#         multiclassResults = ExecClassif.genMetricsScores(cls.multiclassResults, cls.true_labels, cls.metrics)
#         cls.assertEqual(0, multiclassResults[1]["chicken_is_heaven"]["metricsScores"]["accuracy_score"])
#         cls.assertEqual(cls.score_to_get, multiclassResults[0]["chicken_is_heaven"]["metricsScores"]["accuracy_score"])
#
#     def test_all(cls):
#         cls.multiclassResults = [{"chicken_is_heaven": {"labels": cls.multiclass_labels},
#                                                           "cheese_is_no_disease": {"labels": cls.wrong_labels}},
#                                                          {"chicken_is_heaven": {"labels": cls.wrong_labels},
#                                                           "cheese_is_no_disease": {"labels": cls.multiclass_labels}},
#                                                          ]
#         cls.metrics = [["accuracy_score"], ["f1_score"]]
#         from sklearn.metrics import f1_score
#         cls.score_to_get_f1 = f1_score(cls.true_labels, cls.multiclass_labels, average="micro")
#         multiclassResults = ExecClassif.genMetricsScores(cls.multiclassResults, cls.true_labels, cls.metrics)
#         cls.assertEqual(0, multiclassResults[1]["chicken_is_heaven"]["metricsScores"]["accuracy_score"])
#         cls.assertEqual(cls.score_to_get_f1, multiclassResults[1]["cheese_is_no_disease"]["metricsScores"]["f1_score"])
#
#
# class Test_getErrorOnLabels(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.multiclass_labels = np.array([0,1,2,3,4,5,2,1,3])
#         cls.wrong_labels = np.array([1,3,3,4,5,0,2,4,3])
#         cls.multiclassResults = [{"chicken_is_heaven":
#                                       {"labels": cls.multiclass_labels}}]
#         cls.true_labels = np.array([0,2,2,3,4,5,1,3,2])
#
#     def test_simple(cls):
#         multiclassResults = ExecClassif.getErrorOnLabels(cls.multiclassResults, cls.true_labels)
#         np.testing.assert_array_equal(np.array([1, 0, 1, 1, 1, 1, 0, 0, 0]),
#                                       multiclassResults[0]["chicken_is_heaven"]["errorOnExample"])
#
#     def test_full(cls):
#         cls.multiclassResults = [{"chicken_is_heaven": {"labels": cls.multiclass_labels},
#                                   "cheese_is_no_disease": {"labels": cls.wrong_labels}},
#                                  {"chicken_is_heaven": {"labels": cls.wrong_labels},
#                                   "cheese_is_no_disease": {"labels": cls.wrong_labels}},
#                                  ]
#         multiclassResults = ExecClassif.getErrorOnLabels(cls.multiclassResults, cls.true_labels)
#         np.testing.assert_array_equal(np.array([1, 0, 1, 1, 1, 1, 0, 0, 0]),
#                                       multiclassResults[0]["chicken_is_heaven"]["errorOnExample"])
#         np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
#                                       multiclassResults[1]["cheese_is_no_disease"]["errorOnExample"])
#
#     def test_type(cls):
#         multiclassResults = ExecClassif.getErrorOnLabels(cls.multiclassResults, cls.true_labels)
#         cls.assertEqual(type(multiclassResults[0]["chicken_is_heaven"]["errorOnExample"][0]), np.int64)
#         np.testing.assert_array_equal(np.array([1, 0, 1, 1, 1, 1, 0, 0, 0]),
#                                       multiclassResults[0]["chicken_is_heaven"]["errorOnExample"])
#
# class Essai(unittest.TestCase):
#
#     def setUp(self):
#         parser = argparse.ArgumentParser(
#             description='This file is used to benchmark the scores fo multiple classification algorithm on multiview data.',
#             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
#         groupStandard = parser.add_argument_group('Standard arguments')
#         groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
#         groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
#                                    default='Plausible')
#         groupStandard.add_argument('--type', metavar='STRING', action='store',
#                                    help='Type of database : .hdf5 or .csv (default: %(default)s)',
#                                    default='.hdf5')
#         groupStandard.add_argument('--views', metavar='STRING', action='store', nargs="+",
#                                    help='Name of the views selected for learning (default: %(default)s)',
#                                    default=[''])
#         groupStandard.add_argument('--pathF', metavar='STRING', action='store', help='Path to the views (default: %(default)s)',
#                                    default='/home/bbauvin/Documents/data/Data_multi_omics/')
#         groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
#                                    help='Niceness for the process', default=0)
#         groupStandard.add_argument('--random_state', metavar='STRING', action='store',
#                                    help="The random state seed to use or a file where we can find it's get_state", default=None)
#
#         groupClass = parser.add_argument_group('Classification arguments')
#         groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
#                                 help='Determine the split between learning and validation sets', type=float,
#                                 default=0.2)
#         groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
#                                 type=int, default=2)
#         groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
#                                 default=2)
#         groupClass.add_argument('--CL_classes', metavar='STRING', action='store', nargs="+",
#                                 help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
#                                      'selected ex. walrus mole leopard', default=["yes", "no"])
#         groupClass.add_argument('--CL_type', metavar='STRING', action='store', nargs="+",
#                                 help='Determine whether to use multiview and/or monoview, or Benchmark',
#                                 default=['Benchmark'])
#         groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store', nargs="+",
#                                 help='Determine which monoview classifier to use if empty, considering all',
#                                 default=[''])
#         groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store', nargs="+",
#                                 help='Determine which multiview classifier to use if empty, considering all',
#                                 default=[''])
#         groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
#                                 default=2)
#         groupClass.add_argument('--CL_statsiter', metavar='INT', action='store',
#                                 help="Number of iteration for each algorithm to mean preds if using multiple cores, it's highly recommended to use statsiter mod(nbCores) = 0",
#                                 type=int,
#                                 default=2)
#         groupClass.add_argument('--CL_metrics', metavar='STRING', action='store', nargs="+",
#                                 help='Determine which metrics to use, separate metric and configuration with ":".'
#                                      ' If multiple, separate with space. If no metric is specified, '
#                                      'considering all with accuracy for classification '
#                                 , default=[''])
#         groupClass.add_argument('--CL_metric_princ', metavar='STRING', action='store',
#                                 help='Determine which metric to use for randomSearch and optimization', default="f1_score")
#         groupClass.add_argument('--CL_GS_iter', metavar='INT', action='store',
#                                 help='Determine how many Randomized grid search tests to do', type=int, default=2)
#         groupClass.add_argument('--CL_HPS_type', metavar='STRING', action='store',
#                                 help='Determine which hyperparamter search function use', default="randomizedSearch")
#
#         groupRF = parser.add_argument_group('Random Forest arguments')
#         groupRF.add_argument('--CL_RandomForest_trees', metavar='INT', type=int, action='store', help='Number max trees',
#                              default=25)
#         groupRF.add_argument('--CL_RandomForest_max_depth', metavar='INT', type=int, action='store',
#                              help='Max depth for the trees',
#                              default=5)
#         groupRF.add_argument('--CL_RandomForest_criterion', metavar='STRING', action='store', help='Criterion for the trees',
#                              default="entropy")
#
#         groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
#         groupSVMLinear.add_argument('--CL_SVMLinear_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
#                                     default=1)
#
#         groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
#         groupSVMRBF.add_argument('--CL_SVMRBF_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
#                                  default=1)
#
#         groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
#         groupSVMPoly.add_argument('--CL_SVMPoly_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
#                                   default=1)
#         groupSVMPoly.add_argument('--CL_SVMPoly_deg', metavar='INT', type=int, action='store', help='Degree parameter used',
#                                   default=2)
#
#         groupAdaboost = parser.add_argument_group('Adaboost arguments')
#         groupAdaboost.add_argument('--CL_Adaboost_n_est', metavar='INT', type=int, action='store', help='Number of estimators',
#                                    default=2)
#         groupAdaboost.add_argument('--CL_Adaboost_b_est', metavar='STRING', action='store', help='Estimators',
#                                    default='DecisionTreeClassifier')
#
#         groupDT = parser.add_argument_group('Decision Trees arguments')
#         groupDT.add_argument('--CL_DecisionTree_depth', metavar='INT', type=int, action='store',
#                              help='Determine max depth for Decision Trees', default=3)
#         groupDT.add_argument('--CL_DecisionTree_criterion', metavar='STRING', action='store',
#                              help='Determine max depth for Decision Trees', default="entropy")
#         groupDT.add_argument('--CL_DecisionTree_splitter', metavar='STRING', action='store',
#                              help='Determine criterion for Decision Trees', default="random")
#
#         groupSGD = parser.add_argument_group('SGD arguments')
#         groupSGD.add_argument('--CL_SGD_alpha', metavar='FLOAT', type=float, action='store',
#                               help='Determine alpha for SGDClassifier', default=0.1)
#         groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store',
#                               help='Determine loss for SGDClassifier', default='log')
#         groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store',
#                               help='Determine penalty for SGDClassifier', default='l2')
#
#         groupKNN = parser.add_argument_group('KNN arguments')
#         groupKNN.add_argument('--CL_KNN_neigh', metavar='INT', type=int, action='store',
#                               help='Determine number of neighbors for KNN', default=1)
#         groupKNN.add_argument('--CL_KNN_weights', metavar='STRING', action='store',
#                               help='Determine number of neighbors for KNN', default="distance")
#         groupKNN.add_argument('--CL_KNN_algo', metavar='STRING', action='store',
#                               help='Determine number of neighbors for KNN', default="auto")
#         groupKNN.add_argument('--CL_KNN_p', metavar='INT', type=int, action='store',
#                               help='Determine number of neighbors for KNN', default=1)
#
#         groupSCM = parser.add_argument_group('SCM arguments')
#         groupSCM.add_argument('--CL_SCM_max_rules', metavar='INT', type=int, action='store',
#                               help='Max number of rules for SCM', default=1)
#         groupSCM.add_argument('--CL_SCM_p', metavar='FLOAT', type=float, action='store',
#                               help='Max number of rules for SCM', default=1.0)
#         groupSCM.add_argument('--CL_SCM_model_type', metavar='STRING', action='store',
#                               help='Max number of rules for SCM', default="conjunction")
#
#         groupMumbo = parser.add_argument_group('Mumbo arguments')
#         groupMumbo.add_argument('--MU_types', metavar='STRING', action='store', nargs="+",
#                                 help='Determine which monoview classifier to use with Mumbo',
#                                 default=[''])
#         groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
#                                 help='Configuration for the monoview classifier in Mumbo separate each classifier with sapce and each argument with:',
#                                 default=[''])
#         groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
#                                 help='Max number of iteration, min number of iteration, convergence threshold', type=float,
#                                 default=[10, 1, 0.01])
#         groupMumbo.add_argument('--MU_combination', action='store_true',
#                                 help='Try all the monoview classifiers combinations for each view',
#                                 default=False)
#
#
#         groupFusion = parser.add_argument_group('fusion arguments')
#         groupFusion.add_argument('--FU_types', metavar='STRING', action='store', nargs="+",
#                                  help='Determine which type of fusion to use',
#                                  default=[''])
#         groupEarlyFusion = parser.add_argument_group('Early fusion arguments')
#         groupEarlyFusion.add_argument('--FU_early_methods', metavar='STRING', action='store', nargs="+",
#                                       help='Determine which early fusion method of fusion to use',
#                                       default=[''])
#         groupEarlyFusion.add_argument('--FU_E_method_configs', metavar='STRING', action='store', nargs='+',
#                                       help='Configuration for the early fusion methods separate '
#                                            'method by space and values by :',
#                                       default=[''])
#         groupEarlyFusion.add_argument('--FU_E_cl_config', metavar='STRING', action='store', nargs='+',
#                                       help='Configuration for the monoview classifiers used separate classifier by space '
#                                            'and configs must be of form argument1_name:value,argument2_name:value',
#                                       default=[''])
#         groupEarlyFusion.add_argument('--FU_E_cl_names', metavar='STRING', action='store', nargs='+',
#                                       help='Name of the classifiers used for each early fusion method', default=[''])
#
#         groupLateFusion = parser.add_argument_group('Late Early fusion arguments')
#         groupLateFusion.add_argument('--FU_late_methods', metavar='STRING', action='store', nargs="+",
#                                      help='Determine which late fusion method of fusion to use',
#                                      default=[''])
#         groupLateFusion.add_argument('--FU_L_method_config', metavar='STRING', action='store', nargs='+',
#                                      help='Configuration for the fusion method', default=[''])
#         groupLateFusion.add_argument('--FU_L_cl_config', metavar='STRING', action='store', nargs='+',
#                                      help='Configuration for the monoview classifiers used', default=[''])
#         groupLateFusion.add_argument('--FU_L_cl_names', metavar='STRING', action='store', nargs="+",
#                                      help='Names of the classifier used for late fusion', default=[''])
#         groupLateFusion.add_argument('--FU_L_select_monoview', metavar='STRING', action='store',
#                                      help='Determine which method to use to select the monoview classifiers',
#                                      default="intersect")
#         self.args = parser.parse_args([])


# def suite():
#     suite = unittest.TestSuite()
#     suite.addTest(Test_initBenchmark('test_initKWARGSFunc_no_monoview'))
#     # suite.addTest(WidgetTestCase('test_widget_resize'))
#     return suite

if __name__ == '__main__':
    unittest.main()