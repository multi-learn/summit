import os
import unittest

import h5py
import numpy as np

from multiview_platform.tests.utils import rm_tmp, tmp_path, test_dataset

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
                                                            self.init_kwargs,
                                                            "None", {})
        expected_output = [{
                self.monoview_classifier_name: {
                    self.monoview_classifier_arg_name:self.monoview_classifier_arg_value[0]},
                "view_name": "test_view_0",
                'hps_kwargs': {},
                "classifier_name": self.monoview_classifier_name,
                "nb_class": self.nb_class,
                "view_index": 0},
                {self.monoview_classifier_name: {
                    self.monoview_classifier_arg_name: self.monoview_classifier_arg_value[0]},
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
                "view_indices": [0,1],
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                'hps_kwargs': {},
                "labels_names":None,
                self.multiview_classifier_name: {
                    self.multiview_classifier_arg_name:
                        self.multiview_classifier_arg_value[0]},
        },]
        self.assertEqual(arguments["multiview"][0], expected_output[0])

    # def test_init_argument_dictionaries_multiview_multiple(self):
    #     self.multiview_classifier_arg_value = ["fake_value_2", "fake_arg_value_3"]
    #     self.init_kwargs = {
    #         'monoview': {
    #             self.monoview_classifier_name:
    #                 {
    #                     self.monoview_classifier_arg_name: self.monoview_classifier_arg_value}
    #         },
    #         "multiview": {
    #             self.multiview_classifier_name: {
    #                 self.multiview_classifier_arg_name: self.multiview_classifier_arg_value}
    #         }
    #     }
    #     self.benchmark["multiview"] = ["fake_multiview_classifier"]
    #     self.benchmark["monoview"] = {}
    #     arguments = exec_classif.init_argument_dictionaries(self.benchmark,
    #                                                         self.views_dictionnary,
    #                                                         self.nb_class,
    #                                                         self.init_kwargs,
    #                                                         "None", {})
    #     expected_output = [{
    #             "classifier_name": self.multiview_classifier_name+"_fake_value_2",
    #             "view_indices": [0,1],
    #             "view_names": ["test_view_0", "test_view"],
    #             "nb_class": self.nb_class,
    #             'hps_kwargs': {},
    #             "labels_names":None,
    #             self.multiview_classifier_name + "_fake_value_2": {
    #                 self.multiview_classifier_arg_name:
    #                     self.multiview_classifier_arg_value[0]},
    #     },
    #         {
    #             "classifier_name": self.multiview_classifier_name+"_fake_arg_value_3",
    #             "view_indices": [0, 1],
    #             "view_names": ["test_view_0", "test_view"],
    #             "nb_class": self.nb_class,
    #             'hps_kwargs': {},
    #             "labels_names": None,
    #             self.multiview_classifier_name+"_fake_arg_value_3": {
    #                 self.multiview_classifier_arg_name:
    #                     self.multiview_classifier_arg_value[1]},
    #         }
    #     ]
    #     self.assertEqual(arguments["multiview"][0], expected_output[0])

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
                                                            self.init_kwargs,
                                                            "None", {})
        expected_output = [{
                "classifier_name": self.multiview_classifier_name,
                "view_indices": [0,1],
                'hps_kwargs': {},
                "view_names": ["test_view_0", "test_view"],
                "nb_class": self.nb_class,
                "labels_names":None,
                self.multiview_classifier_name: {
                    self.multiview_classifier_arg_name:
                        self.multiview_classifier_arg_value},
        }]
        self.assertEqual(arguments["multiview"][0], expected_output[0])

    # def test_init_argument_dictionaries_multiview_multiple_complex(self):
    #     self.multiview_classifier_arg_value = {"fake_value_2":["plif", "pluf"], "plaf":"plouf"}
    #     self.init_kwargs = {
    #         'monoview': {
    #             self.monoview_classifier_name:
    #                 {
    #                     self.monoview_classifier_arg_name: self.monoview_classifier_arg_value}
    #         },
    #         "multiview": {
    #             self.multiview_classifier_name: {
    #                 self.multiview_classifier_arg_name: self.multiview_classifier_arg_value}
    #         }
    #     }
    #     self.benchmark["multiview"] = ["fake_multiview_classifier"]
    #     self.benchmark["monoview"] = {}
    #     arguments = exec_classif.init_argument_dictionaries(self.benchmark,
    #                                                         self.views_dictionnary,
    #                                                         self.nb_class,
    #                                                         self.init_kwargs,
    #                                                         "None", {})
    #     expected_output = [{
    #             "classifier_name": self.multiview_classifier_name+"_plif_plouf",
    #             "view_indices": [0,1],
    #             "view_names": ["test_view_0", "test_view"],
    #             "nb_class": self.nb_class,
    #             "labels_names":None,
    #             'hps_kwargs': {},
    #             self.multiview_classifier_name + "_plif_plouf": {
    #                 self.multiview_classifier_arg_name:
    #                     {"fake_value_2": "plif", "plaf": "plouf"}},
    #     },
    #         {
    #             "classifier_name": self.multiview_classifier_name+"_pluf_plouf",
    #             "view_indices": [0, 1],
    #             "view_names": ["test_view_0", "test_view"],
    #             "nb_class": self.nb_class,
    #             "labels_names": None,
    #             'hps_kwargs': {},
    #             self.multiview_classifier_name+"_pluf_plouf": {
    #                 self.multiview_classifier_arg_name:
    #                     {"fake_value_2":"pluf", "plaf":"plouf"}},
    #         }
    #     ]
    #     self.assertEqual(arguments["multiview"][0], expected_output[0])


def fakeBenchmarkExec(core_index=-1, a=7, args=1):
    return [core_index, a]


def fakeBenchmarkExec_mutlicore(nb_cores=-1, a=6, args=1):
    return [nb_cores, a]


def fakeBenchmarkExec_monocore(dataset_var=1, a=4, args=1, track_tracebacks=False):
    return [a]


def fakegetResults(results, stats_iter,
                   benchmark_arguments_dictionaries, metrics, directory,
                   example_ids, labels):
    return 3


def fakeDelete(a, b, c):
    return 9

def fake_analyze(a, b, c, d, example_ids=None, labels=None):
    pass

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
                                          get_results=fakegetResults,
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
                                         get_results=fakegetResults,
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
                                         get_results=fakegetResults,
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
                                         get_results=fakegetResults,
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


# class Test_execOneBenchmark(unittest.TestCase):
#
#     @classmethod
#     def setUp(cls):
#         rm_tmp()
#         os.mkdir(tmp_path)
#         cls.args = {
#             "Base": {"name": "chicken_is_heaven", "type": "type",
#                      "pathf": "pathF"},
#             "Classification": {"hps_iter": 1}}
#
#     def test_simple(cls):
#         flag, results = exec_classif.exec_one_benchmark(core_index=10,
#                                                       labels_dictionary={
#                                                                    0: "a",
#                                                                    1: "b"},
#                                                       directory=tmp_path,
#                                                       classification_indices=(
#                                                                [1, 2, 3, 4],
#                                                                [0, 5, 6, 7, 8]),
#                                                                args=cls.args,
#                                                                k_folds=FakeKfold(),
#                                                                random_state="try",
#                                                                hyper_param_search="try",
#                                                                metrics="try",
#                                                                argument_dictionaries={
#                                                                    "Monoview": [
#                                                                        {
#                                                                            "try": 0},
#                                                                        {
#                                                                            "try2": 100}],
#                                                                    "multiview":[{
#                                                                            "try3": 5},
#                                                                        {
#                                                                            "try4": 10}]},
#                                                       benchmark="try",
#                                                       views="try",
#                                                       views_indices="try",
#                                                       flag=None,
#                                                       labels=np.array(
#                                                                    [0, 1, 2, 1,
#                                                                     2, 2, 2, 12,
#                                                                     1, 2, 1, 1,
#                                                                     2, 1, 21]),
#                                                       exec_monoview_multicore=fakeExecMono,
#                                                       exec_multiview_multicore=fakeExecMulti,)
#
#         cls.assertEqual(flag, None)
#         cls.assertEqual(results ,
#                         [['Mono', {'try': 0}], ['Mono', {'try2': 100}],
#                          ['Multi', {'try3': 5}], ['Multi', {'try4': 10}]])
#
#     @classmethod
#     def tearDown(cls):
#         path = tmp_path
#         for file_name in os.listdir(path):
#             dir_path = os.path.join(path, file_name)
#             if os.path.isdir(dir_path):
#                 for file_name in os.listdir(dir_path):
#                     os.remove(os.path.join(dir_path, file_name))
#                 os.rmdir(dir_path)
#             else:
#                 os.remove(os.path.join(path, file_name))
#         os.rmdir(path)
#
#
# class Test_execOneBenchmark_multicore(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         rm_tmp()
#         os.mkdir(tmp_path)
#         cls.args = {
#             "Base": {"name": "chicken_is_heaven", "type": "type",
#                      "pathf": "pathF"},
#             "Classification": {"hps_iter": 1}}
#
#     def test_simple(cls):
#         flag, results = exec_classif.exec_one_benchmark_multicore(
#             nb_cores=2,
#             labels_dictionary={0: "a", 1: "b"},
#             directory=tmp_path,
#             classification_indices=([1, 2, 3, 4], [0, 10, 20, 30, 40]),
#             args=cls.args,
#             k_folds=FakeKfold(),
#             random_state="try",
#             hyper_param_search="try",
#             metrics="try",
#             argument_dictionaries={
#                                                                    "monoview": [
#                                                                        {
#                                                                            "try": 0},
#                                                                        {
#                                                                            "try2": 100}],
#                                                                    "multiview":[{
#                                                                            "try3": 5},
#                                                                        {
#                                                                            "try4": 10}]},
#             benchmark="try",
#             views="try",
#             views_indices="try",
#             flag=None,
#             labels=np.array([0, 1, 2, 3, 4, 2, 2, 12, 1, 2, 1, 1, 2, 1, 21]),
#             exec_monoview_multicore=fakeExecMono,
#             exec_multiview_multicore=fakeExecMulti,)
#
#         cls.assertEqual(flag, None)
#         cls.assertEqual(results ,
#                         [['Mono', {'try': 0}], ['Mono', {'try2': 100}],
#                          ['Multi', {'try3': 5}], ['Multi', {'try4': 10}]])
#
#     @classmethod
#     def tearDown(cls):
#         path = tmp_path
#         for file_name in os.listdir(path):
#             dir_path = os.path.join(path, file_name)
#             if os.path.isdir(dir_path):
#                 for file_name in os.listdir(dir_path):
#                     os.remove(os.path.join(dir_path, file_name))
#                 os.rmdir(dir_path)
#             else:
#                 os.remove(os.path.join(path, file_name))
#         os.rmdir(path)


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



if __name__ == '__main__':
    unittest.main()