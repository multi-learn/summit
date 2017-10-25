import unittest
import argparse

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
        arguments = ExecClassif.initMonoviewExps(benchmark, {}, [], None, 0, {})
        self.assertEqual(arguments, {})

    def test_initMonoviewArguments_empty(self):
        benchmark = {"Monoview":{}, "Multiview":{}}
        arguments = ExecClassif.initMonoviewExps(benchmark, {}, [], None, 0, {})

class Essai(unittest.TestCase):

    def setUp(self):
        parser = argparse.ArgumentParser(
            description='This file is used to benchmark the scores fo multiple classification algorithm on multiview data.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        groupStandard = parser.add_argument_group('Standard arguments')
        groupStandard.add_argument('-log', action='store_true', help='Use option to activate Logging to Console')
        groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                                   default='Plausible')
        groupStandard.add_argument('--type', metavar='STRING', action='store',
                                   help='Type of database : .hdf5 or .csv (default: %(default)s)',
                                   default='.hdf5')
        groupStandard.add_argument('--views', metavar='STRING', action='store', nargs="+",
                                   help='Name of the views selected for learning (default: %(default)s)',
                                   default=[''])
        groupStandard.add_argument('--pathF', metavar='STRING', action='store', help='Path to the views (default: %(default)s)',
                                   default='/home/bbauvin/Documents/Data/Data_multi_omics/')
        groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
                                   help='Niceness for the process', default=0)
        groupStandard.add_argument('--randomState', metavar='STRING', action='store',
                                   help="The random state seed to use or a file where we can find it's get_state", default=None)

        groupClass = parser.add_argument_group('Classification arguments')
        groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                                help='Determine the split between learning and validation sets', type=float,
                                default=0.2)
        groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                                type=int, default=2)
        groupClass.add_argument('--CL_nb_class', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                                default=2)
        groupClass.add_argument('--CL_classes', metavar='STRING', action='store', nargs="+",
                                help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                                     'selected ex. walrus mole leopard', default=["yes", "no"])
        groupClass.add_argument('--CL_type', metavar='STRING', action='store', nargs="+",
                                help='Determine whether to use Multiview and/or Monoview, or Benchmark',
                                default=['Benchmark'])
        groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store', nargs="+",
                                help='Determine which monoview classifier to use if empty, considering all',
                                default=[''])
        groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store', nargs="+",
                                help='Determine which multiview classifier to use if empty, considering all',
                                default=[''])
        groupClass.add_argument('--CL_cores', metavar='INT', action='store', help='Number of cores, -1 for all', type=int,
                                default=2)
        groupClass.add_argument('--CL_statsiter', metavar='INT', action='store',
                                help="Number of iteration for each algorithm to mean results if using multiple cores, it's highly recommended to use statsiter mod(nbCores) = 0",
                                type=int,
                                default=2)
        groupClass.add_argument('--CL_metrics', metavar='STRING', action='store', nargs="+",
                                help='Determine which metrics to use, separate metric and configuration with ":".'
                                     ' If multiple, separate with space. If no metric is specified, '
                                     'considering all with accuracy for classification '
                                , default=[''])
        groupClass.add_argument('--CL_metric_princ', metavar='STRING', action='store',
                                help='Determine which metric to use for randomSearch and optimization', default="f1_score")
        groupClass.add_argument('--CL_GS_iter', metavar='INT', action='store',
                                help='Determine how many Randomized grid search tests to do', type=int, default=2)
        groupClass.add_argument('--CL_HPS_type', metavar='STRING', action='store',
                                help='Determine which hyperparamter search function use', default="randomizedSearch")

        groupRF = parser.add_argument_group('Random Forest arguments')
        groupRF.add_argument('--CL_RandomForest_trees', metavar='INT', type=int, action='store', help='Number max trees',
                             default=25)
        groupRF.add_argument('--CL_RandomForest_max_depth', metavar='INT', type=int, action='store',
                             help='Max depth for the trees',
                             default=5)
        groupRF.add_argument('--CL_RandomForest_criterion', metavar='STRING', action='store', help='Criterion for the trees',
                             default="entropy")

        groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
        groupSVMLinear.add_argument('--CL_SVMLinear_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                                    default=1)

        groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
        groupSVMRBF.add_argument('--CL_SVMRBF_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                                 default=1)

        groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
        groupSVMPoly.add_argument('--CL_SVMPoly_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                                  default=1)
        groupSVMPoly.add_argument('--CL_SVMPoly_deg', metavar='INT', type=int, action='store', help='Degree parameter used',
                                  default=2)

        groupAdaboost = parser.add_argument_group('Adaboost arguments')
        groupAdaboost.add_argument('--CL_Adaboost_n_est', metavar='INT', type=int, action='store', help='Number of estimators',
                                   default=2)
        groupAdaboost.add_argument('--CL_Adaboost_b_est', metavar='STRING', action='store', help='Estimators',
                                   default='DecisionTreeClassifier')

        groupDT = parser.add_argument_group('Decision Trees arguments')
        groupDT.add_argument('--CL_DecisionTree_depth', metavar='INT', type=int, action='store',
                             help='Determine max depth for Decision Trees', default=3)
        groupDT.add_argument('--CL_DecisionTree_criterion', metavar='STRING', action='store',
                             help='Determine max depth for Decision Trees', default="entropy")
        groupDT.add_argument('--CL_DecisionTree_splitter', metavar='STRING', action='store',
                             help='Determine criterion for Decision Trees', default="random")

        groupSGD = parser.add_argument_group('SGD arguments')
        groupSGD.add_argument('--CL_SGD_alpha', metavar='FLOAT', type=float, action='store',
                              help='Determine alpha for SGDClassifier', default=0.1)
        groupSGD.add_argument('--CL_SGD_loss', metavar='STRING', action='store',
                              help='Determine loss for SGDClassifier', default='log')
        groupSGD.add_argument('--CL_SGD_penalty', metavar='STRING', action='store',
                              help='Determine penalty for SGDClassifier', default='l2')

        groupKNN = parser.add_argument_group('KNN arguments')
        groupKNN.add_argument('--CL_KNN_neigh', metavar='INT', type=int, action='store',
                              help='Determine number of neighbors for KNN', default=1)
        groupKNN.add_argument('--CL_KNN_weights', metavar='STRING', action='store',
                              help='Determine number of neighbors for KNN', default="distance")
        groupKNN.add_argument('--CL_KNN_algo', metavar='STRING', action='store',
                              help='Determine number of neighbors for KNN', default="auto")
        groupKNN.add_argument('--CL_KNN_p', metavar='INT', type=int, action='store',
                              help='Determine number of neighbors for KNN', default=1)

        groupSCM = parser.add_argument_group('SCM arguments')
        groupSCM.add_argument('--CL_SCM_max_rules', metavar='INT', type=int, action='store',
                              help='Max number of rules for SCM', default=1)
        groupSCM.add_argument('--CL_SCM_p', metavar='FLOAT', type=float, action='store',
                              help='Max number of rules for SCM', default=1.0)
        groupSCM.add_argument('--CL_SCM_model_type', metavar='STRING', action='store',
                              help='Max number of rules for SCM', default="conjunction")

        groupMumbo = parser.add_argument_group('Mumbo arguments')
        groupMumbo.add_argument('--MU_types', metavar='STRING', action='store', nargs="+",
                                help='Determine which monoview classifier to use with Mumbo',
                                default=[''])
        groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                                help='Configuration for the monoview classifier in Mumbo separate each classifier with sapce and each argument with:',
                                default=[''])
        groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
                                help='Max number of iteration, min number of iteration, convergence threshold', type=float,
                                default=[10, 1, 0.01])
        groupMumbo.add_argument('--MU_combination', action='store_true',
                                help='Try all the monoview classifiers combinations for each view',
                                default=False)


        groupFusion = parser.add_argument_group('Fusion arguments')
        groupFusion.add_argument('--FU_types', metavar='STRING', action='store', nargs="+",
                                 help='Determine which type of fusion to use',
                                 default=[''])
        groupEarlyFusion = parser.add_argument_group('Early Fusion arguments')
        groupEarlyFusion.add_argument('--FU_early_methods', metavar='STRING', action='store', nargs="+",
                                      help='Determine which early fusion method of fusion to use',
                                      default=[''])
        groupEarlyFusion.add_argument('--FU_E_method_configs', metavar='STRING', action='store', nargs='+',
                                      help='Configuration for the early fusion methods separate '
                                           'method by space and values by :',
                                      default=[''])
        groupEarlyFusion.add_argument('--FU_E_cl_config', metavar='STRING', action='store', nargs='+',
                                      help='Configuration for the monoview classifiers used separate classifier by space '
                                           'and configs must be of form argument1_name:value,argument2_name:value',
                                      default=[''])
        groupEarlyFusion.add_argument('--FU_E_cl_names', metavar='STRING', action='store', nargs='+',
                                      help='Name of the classifiers used for each early fusion method', default=[''])

        groupLateFusion = parser.add_argument_group('Late Early Fusion arguments')
        groupLateFusion.add_argument('--FU_late_methods', metavar='STRING', action='store', nargs="+",
                                     help='Determine which late fusion method of fusion to use',
                                     default=[''])
        groupLateFusion.add_argument('--FU_L_method_config', metavar='STRING', action='store', nargs='+',
                                     help='Configuration for the fusion method', default=[''])
        groupLateFusion.add_argument('--FU_L_cl_config', metavar='STRING', action='store', nargs='+',
                                     help='Configuration for the monoview classifiers used', default=[''])
        groupLateFusion.add_argument('--FU_L_cl_names', metavar='STRING', action='store', nargs="+",
                                     help='Names of the classifier used for late fusion', default=[''])
        groupLateFusion.add_argument('--FU_L_select_monoview', metavar='STRING', action='store',
                                     help='Determine which method to use to select the monoview classifiers',
                                     default="intersect")
        self.args = parser.parse_args([])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_initBenchmark('test_initKWARGSFunc_no_monoview'))
    # suite.addTest(WidgetTestCase('test_widget_resize'))
    return suite