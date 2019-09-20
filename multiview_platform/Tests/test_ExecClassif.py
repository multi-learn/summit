import os
import unittest

import h5py
import numpy as np

from ..MonoMultiViewClassifiers import ExecClassif


class Test_initBenchmark(unittest.TestCase):

    def test_benchmark_wanted(self):
        # benchmark_output = ExecClassif.initBenchmark(self.args)
        self.assertEqual(1, 1)


class Test_initKWARGS(unittest.TestCase):

    def test_initKWARGSFunc_no_monoview(self):
        benchmark = {"Monoview": {}, "Multiview": {}}
        args = ExecClassif.initKWARGSFunc({}, benchmark)
        self.assertEqual(args, {})


class Test_initMonoviewArguments(unittest.TestCase):

    def test_initMonoviewArguments_no_monoview(self):
        benchmark = {"Monoview": {}, "Multiview": {}}
        arguments = ExecClassif.initMonoviewExps(benchmark, {}, 0, {})
        self.assertEqual(arguments, {'Monoview': [], 'Multiview': []})

    def test_initMonoviewArguments_empty(self):
        benchmark = {"Monoview": {}, "Multiview": {}}
        arguments = ExecClassif.initMonoviewExps(benchmark, {}, 0, {})


def fakeBenchmarkExec(coreIndex=-1, a=7, args=1):
    return [coreIndex, a]


def fakeBenchmarkExec_mutlicore(nbCores=-1, a=6, args=1):
    return [nbCores, a]


def fakeBenchmarkExec_monocore(DATASET=1, a=4, args=1):
    return [a]


def fakegetResults(results, statsIter, nbMulticlass,
                   benchmarkArgumentsDictionaries, multiClassLabels, metrics,
                   classificationIndices, directories, directory,
                   labelsDictionary, nbExamples, nbLabels):
    return 3


def fakeDelete(a, b, c):
    return 9


class Test_execBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.mkdir("multiview_platform/Tests/tmp_tests")
        cls.Dataset = h5py.File(
            "multiview_platform/Tests/tmp_tests/test_file.hdf5", "w")
        cls.labels = cls.Dataset.create_dataset("Labels",
                                                data=np.array([0, 1, 2]))
        cls.argumentDictionaries = [{"a": 4, "args": {}}]
        cls.args = {
            "Base":{"name": "chicken_is_heaven", "type": "type", "pathf": "pathF"},
            "Classification":{"hps_iter": 1}}

    def test_simple(cls):
        res = ExecClassif.execBenchmark(1, 2, 3, cls.argumentDictionaries,
                                        [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                        10, cls.Dataset,
                                        execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore,
                                        getResults=fakegetResults,
                                        delete=fakeDelete)
        cls.assertEqual(res, 3)

    def test_multiclass_no_iter(cls):
        cls.argumentDictionaries = [{"a": 10, "args": cls.args},
                                    {"a": 4, "args":cls.args}]
        res = ExecClassif.execBenchmark(2, 1, 2, cls.argumentDictionaries,
                                        [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                        10, cls.Dataset,
                                        execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore,
                                        getResults=fakegetResults,
                                        delete=fakeDelete)
        cls.assertEqual(res, 3)

    def test_multiclass_and_iter(cls):
        cls.argumentDictionaries = [{"a": 10, "args": cls.args},
                                    {"a": 4, "args": cls.args},
                                    {"a": 55, "args": cls.args},
                                    {"a": 24, "args": cls.args}]
        res = ExecClassif.execBenchmark(2, 2, 2, cls.argumentDictionaries,
                                        [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                        10, cls.Dataset,
                                        execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore,
                                        getResults=fakegetResults,
                                        delete=fakeDelete)
        cls.assertEqual(res, 3)

    def test_no_iter_biclass_multicore(cls):
        res = ExecClassif.execBenchmark(2, 1, 1, cls.argumentDictionaries,
                                        [[[1, 2], [3, 4, 5]]], 5, 6, 7, 8, 9,
                                        10, cls.Dataset,
                                        execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore,
                                        getResults=fakegetResults,
                                        delete=fakeDelete)
        cls.assertEqual(res, 3)

    @classmethod
    def tearDownClass(cls):
        cls.Dataset.close()
        path = "multiview_platform/Tests/tmp_tests/"
        for file_name in os.listdir(path):
            os.remove(os.path.join(path, file_name))
        os.rmdir(path)


def fakeExecMono(directory, name, labelsNames, classificationIndices, kFolds,
                 coreIndex, type, pathF, randomState, labels,
                 hyperParamSearch="try", metrics="try", nIter=1, **arguments):
    return ["Mono", arguments]


def fakeExecMulti(directory, coreIndex, name, classificationIndices, kFolds,
                  type, pathF, LABELS_DICTIONARY,
                  randomState, labels, hyperParamSearch="", metrics=None,
                  nIter=1, **arguments):
    return ["Multi", arguments]


def fakeInitMulti(args, benchmark, views, viewsIndices, argumentDictionaries,
                  randomState, directory, resultsMonoview,
                  classificationIndices):
    return {"Monoview": [{"try": 0}, {"try2": 100}],
            "Multiview": [{"try3": 5}, {"try4": 10}]}


class FakeKfold():
    def __init__(self):
        self.n_splits = 2
        pass

    def split(self, X, Y):
        return [([X[0], X[1]], [X[2], X[3]]), (([X[2], X[3]], [X[0], X[1]]))]


class Test_execOneBenchmark(unittest.TestCase):

    @classmethod
    def setUp(cls):
        os.mkdir("multiview_platform/Tests/tmp_tests")
        cls.args = {
            "Base": {"name": "chicken_is_heaven", "type": "type",
                     "pathf": "pathF"},
            "Classification": {"hps_iter": 1}}

    def test_simple(cls):
        flag, results = ExecClassif.execOneBenchmark(coreIndex=10,
                                                               LABELS_DICTIONARY={
                                                                   0: "a",
                                                                   1: "b"},
                                                               directory="multiview_platform/Tests/tmp_tests/",
                                                               classificationIndices=(
                                                               [1, 2, 3, 4],
                                                               [0, 5, 6, 7, 8]),
                                                               args=cls.args,
                                                               kFolds=FakeKfold(),
                                                               randomState="try",
                                                               hyperParamSearch="try",
                                                               metrics="try",
                                                               argumentDictionaries={
                                                                   "Monoview": [
                                                                       {
                                                                           "try": 0},
                                                                       {
                                                                           "try2": 100}]},
                                                               benchmark="try",
                                                               views="try",
                                                               viewsIndices="try",
                                                               flag=None,
                                                               labels=np.array(
                                                                   [0, 1, 2, 1,
                                                                    2, 2, 2, 12,
                                                                    1, 2, 1, 1,
                                                                    2, 1, 21]),
                                                               ExecMonoview_multicore=fakeExecMono,
                                                               ExecMultiview_multicore=fakeExecMulti,
                                                               initMultiviewArguments=fakeInitMulti)

        cls.assertEqual(flag, None)
        cls.assertEqual(results ,
                        [['Mono', {'try': 0}], ['Mono', {'try2': 100}],
                         ['Multi', {'try3': 5}], ['Multi', {'try4': 10}]])

    @classmethod
    def tearDown(cls):
        path = "multiview_platform/Tests/tmp_tests/"
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
        os.mkdir("multiview_platform/Tests/tmp_tests")
        cls.args = {
            "Base": {"name": "chicken_is_heaven", "type": "type",
                     "pathf": "pathF"},
            "Classification": {"hps_iter": 1}}

    def test_simple(cls):
        flag, results = ExecClassif.execOneBenchmark_multicore(
            nbCores=2,
            LABELS_DICTIONARY={0: "a", 1: "b"},
            directory="multiview_platform/Tests/tmp_tests/",
            classificationIndices=([1, 2, 3, 4], [0, 10, 20, 30, 40]),
            args=cls.args,
            kFolds=FakeKfold(),
            randomState="try",
            hyperParamSearch="try",
            metrics="try",
            argumentDictionaries={"Monoview": [{"try": 0}, {"try2": 100}]},
            benchmark="try",
            views="try",
            viewsIndices="try",
            flag=None,
            labels=np.array([0, 1, 2, 3, 4, 2, 2, 12, 1, 2, 1, 1, 2, 1, 21]),
            ExecMonoview_multicore=fakeExecMono,
            ExecMultiview_multicore=fakeExecMulti,
            initMultiviewArguments=fakeInitMulti)

        cls.assertEqual(flag, None)
        cls.assertEqual(results ,
                        [['Mono', {'try': 0}], ['Mono', {'try2': 100}],
                         ['Multi', {'try3': 5}], ['Multi', {'try4': 10}]])

    @classmethod
    def tearDown(cls):
        path = "multiview_platform/Tests/tmp_tests/"
        for file_name in os.listdir(path):
            dir_path = os.path.join(path, file_name)
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file_name))
                os.rmdir(dir_path)
            else:
                os.remove(os.path.join(path, file_name))
        os.rmdir(path)

