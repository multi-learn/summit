import unittest
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score

from ..MonoMultiViewClassifiers import ExecClassif




class Test_initBenchmark(unittest.TestCase):

    def test_benchmark_wanted(self):
        # benchmark_output = ExecClassif.initBenchmark(self.args)
        self.assertEqual(1,1)


class Test_initKWARGS(unittest.TestCase):

    def test_initKWARGSFunc_no_monoview(self):
        benchmark = {"Monoview":{}, "Multiview":{}}
        args = ExecClassif.initKWARGSFunc({}, benchmark)
        self.assertEqual(args, {})


class Test_initMonoviewArguments(unittest.TestCase):

    def test_initMonoviewArguments_no_monoview(self):
        benchmark = {"Monoview":{}, "Multiview":{}}
        arguments = ExecClassif.initMonoviewExps(benchmark, {}, {}, 0, {})
        self.assertEqual(arguments, {})

    def test_initMonoviewArguments_empty(self):
        benchmark = {"Monoview":{}, "Multiview":{}}
        arguments = ExecClassif.initMonoviewExps(benchmark, {}, {}, 0, {})


def fakeBenchmarkExec(coreIndex=-1, a=7):
    return [coreIndex, a]


def fakeBenchmarkExec_mutlicore(nbCores=-1, a=6):
    return [nbCores,a]

def fakeBenchmarkExec_monocore(a=4):
    return a


class Test_execBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.argumentDictionaries = [{"a": 4}]

    def test_simple(cls):
        res = ExecClassif.execBenchmark(1,1,1,cls.argumentDictionaries, execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore)
        cls.assertEqual(res, [4])

    def test_multiclass_no_iter(cls):
        cls.argumentDictionaries = [{"a": 10}, {"a": 4}]
        res = ExecClassif.execBenchmark(2,1,2,cls.argumentDictionaries, execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore)
        cls.assertEqual(res, [[0,10], [1,4]])

    def test_multiclass_and_iter(cls):
        cls.argumentDictionaries = [{"a": 10}, {"a": 4}, {"a": 55}, {"a": 24}]
        res = ExecClassif.execBenchmark(2,2,2,cls.argumentDictionaries, execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore)
        cls.assertEqual(res, [[0,10], [1,4], [0,55], [1,24]])

    def test_no_iter_biclass_multicore(cls):
        res = ExecClassif.execBenchmark(2,1,1,cls.argumentDictionaries, execOneBenchmark=fakeBenchmarkExec,
                                        execOneBenchmark_multicore=fakeBenchmarkExec_mutlicore,
                                        execOneBenchmarkMonoCore=fakeBenchmarkExec_monocore)
        cls.assertEqual(res, [[2,4]])


def fakeExecMono(directory, name, labelsNames, classificationIndices, kFolds, coreIndex, type, pathF, randomState, labels,
                 hyperParamSearch="try", metrics="try", nIter=1, **arguments):
    return ["Mono", arguments]


def fakeExecMulti(directory, coreIndex, name, classificationIndices, kFolds, type, pathF, LABELS_DICTIONARY,
                  randomState, labels, hyperParamSearch="",metrics=None, nIter=1, **arguments):
    return ["Multi", arguments]


def fakeInitMulti(args, benchmark, views, viewsIndices, argumentDictionaries,
                  randomState, directory, resultsMonoview, classificationIndices):
    return {"Monoview": [{"try": 0}, {"try2": 100}], "Multiview": [{"try3": 5}, {"try4": 10}]}


class FakeArg(object):
    def __init__(self):
        self.name = "chicken_is_heaven"
        self.type = "type"
        self.pathF = "pathF"
        self.CL_GS_iter = 1


class Test_execOneBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        os.mkdir("Code/Tests/tmp_tests")

    def test_simple(cls):
        flag, resMono, resMulti = ExecClassif.execOneBenchmark(coreIndex=10,
                                                               LABELS_DICTIONARY={0: "a", 1: "b"},
                                                               directory="Code/Tests/tmp_tests/",
                                                               classificationIndices=([1,2,3,4], [0,5,6,7,8]),
                                                               args=FakeArg(),
                                                               kFolds="try",
                                                               randomState="try",
                                                               hyperParamSearch="try",
                                                               metrics="try",
                                                               argumentDictionaries={"Monoview": [{"try": 0}, {"try2": 100}]},
                                                               benchmark="try",
                                                               views="try",
                                                               viewsIndices="try",
                                                               flag=None,
                                                               labels=np.array([0,1,2,3,4,2,2,12,1,2,1,1,2,1,21]),
                                                               ExecMonoview_multicore=fakeExecMono,
                                                               ExecMultiview_multicore=fakeExecMulti,
                                                               initMultiviewArguments=fakeInitMulti)

        cls.assertEqual(flag, None)
        cls.assertEqual(resMono, [["Mono", {"try": 0}], ["Mono", {"try2": 100}]])
        cls.assertEqual(resMulti, [["Multi", {"try3": 5}], ["Multi", {"try4": 10}]])

    @classmethod
    def tearDownClass(cls):
        os.remove("Code/Tests/tmp_tests/train_indices.csv")
        os.remove("Code/Tests/tmp_tests/train_labels.csv")
        os.rmdir("Code/Tests/tmp_tests")


class Test_execOneBenchmark_multicore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.mkdir("Code/Tests/tmp_tests")

    def test_simple(cls):
        flag, resMono, resMulti = ExecClassif.execOneBenchmark_multicore(nbCores=2,
                                                               LABELS_DICTIONARY={0: "a", 1: "b"},
                                                               directory="Code/Tests/tmp_tests/",
                                                               classificationIndices=([1,2,3,4], [0,10,20,30,40]),
                                                               args=FakeArg(),
                                                               kFolds="try",
                                                               randomState="try",
                                                               hyperParamSearch="try",
                                                               metrics="try",
                                                               argumentDictionaries={"Monoview": [{"try": 0}, {"try2": 100}]},
                                                               benchmark="try",
                                                               views="try",
                                                               viewsIndices="try",
                                                               flag=None,
                                                               labels=np.array([0,1,2,3,4,2,2,12,1,2,1,1,2,1,21]),
                                                               ExecMonoview_multicore=fakeExecMono,
                                                               ExecMultiview_multicore=fakeExecMulti,
                                                               initMultiviewArguments=fakeInitMulti)

        cls.assertEqual(flag, None)
        cls.assertEqual(resMono, [["Mono", {"try": 0}], ["Mono", {"try2": 100}]])
        cls.assertEqual(resMulti, [["Multi", {"try3": 5}], ["Multi", {"try4": 10}]])

    @classmethod
    def tearDownClass(cls):
        os.remove("Code/Tests/tmp_tests/train_indices.csv")
        os.remove("Code/Tests/tmp_tests/train_labels.csv")
        os.rmdir("Code/Tests/tmp_tests")
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
#         cls.classifiersNames = ["chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven",
#                                 "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven",
#                                 "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven", "chicken_is_heaven",]
#         cls.classifiersNames2 = ["cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease",
#                                 "cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease",
#                                 "cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease",
#                                 "cheese_is_no_disease", "cheese_is_no_disease", "cheese_is_no_disease"]
#         cls.results = [[flag, [["", [name, "", "", pred]], ["", [name1, "", "", pred1]]], ["", ""]]
#                        for flag, name, pred, name1, pred1 in zip(cls.flags, cls.classifiersNames, cls.preds,
#                                                                  cls.classifiersNames2, cls.preds2)]
#         # cls.results = [[flag, ["", ["", name, "", pred]], ""] for flag, pred, name in
#         #                zip(cls.flags, cls.preds, cls.classifiersNames)]
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
#                                    default='/home/bbauvin/Documents/Data/Data_multi_omics/')
#         groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
#                                    help='Niceness for the process', default=0)
#         groupStandard.add_argument('--randomState', metavar='STRING', action='store',
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
#                                 help='Determine whether to use Multiview and/or Monoview, or Benchmark',
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
#         groupFusion = parser.add_argument_group('Fusion arguments')
#         groupFusion.add_argument('--FU_types', metavar='STRING', action='store', nargs="+",
#                                  help='Determine which type of fusion to use',
#                                  default=[''])
#         groupEarlyFusion = parser.add_argument_group('Early Fusion arguments')
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
#         groupLateFusion = parser.add_argument_group('Late Early Fusion arguments')
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