import argparse
import logging
import os
import pickle
import time

import numpy as np
import sklearn

from . import get_multiview_db as DB


def parse_the_args(arguments):
    """Used to parse the args entered by the user"""

    parser = argparse.ArgumentParser(
        description='This file is used to benchmark the scores fo multiple '
                    'classification algorithm on multiview data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('--path_config', metavar='STRING', action='store',
                               help='Path to the hdf5 dataset or database '
                                    'folder (default: %(default)s)',
                               default='../config_files/config.yml')
#     groupStandard.add_argument('-log', action='store_true',
#                                help='Use option to activate logging to console')
#     groupStandard.add_argument('--name', metavar='STRING', nargs='+', action='store',
#                                help='Name of Database (default: %(default)s)',
#                                default=['Plausible'])
#     groupStandard.add_argument('--label', metavar='STRING', action='store',
#                                help='Labeling the results directory (default: '
#                                     '%(default)s)',
#                                default='')
#     groupStandard.add_argument('--type', metavar='STRING', action='store',
#                                help='Type of database : .hdf5 or .csv ('
#                                     'default: %(default)s)',
#                                default='.hdf5')
#     groupStandard.add_argument('--views', metavar='STRING', action='store',
#                                nargs="+",
#                                help='Name of the views selected for learning '
#                                     '(default: %(default)s)',
#                                default=[''])
#     groupStandard.add_argument('--pathF', metavar='STRING', action='store',
#                                help='Path to the hdf5 dataset or database '
#                                     'folder (default: %(default)s)',
#                                default='../data/')
#     groupStandard.add_argument('--nice', metavar='INT', action='store',
#                                type=int,
#                                help='Niceness for the processes', default=0)
#     groupStandard.add_argument('--randomState', metavar='STRING',
#                                action='store',
#                                help="The random state seed to use or the path "
#                                     "to a pickle file where it is stored",
#                                default=None)
#     groupStandard.add_argument('--nbCores', metavar='INT', action='store',
#                                help='Number of cores to use for parallel '
#                                     'computing, -1 for all',
#                                type=int, default=2)
#     groupStandard.add_argument('--machine', metavar='STRING', action='store',
#                                help='Type of machine on which the script runs',
#                                default="PC")
#     groupStandard.add_argument('-full', action='store_true',
#                                help='Use option to use full dataset and no '
#                                     'labels or view filtering')
#     groupStandard.add_argument('-debug', action='store_true',
#                                help='Use option to bebug implemented algorithms')
#     groupStandard.add_argument('-add_noise', action='store_true',
#                                help='Use option to add noise to the data')
#     groupStandard.add_argument('--noise_std', metavar='FLOAT', nargs="+", action='store',
#                                help='The std of the gaussian noise that will '
#                                     'be added to the data.',
#                                type=float, default=[0.0])
#     groupStandard.add_argument('--res_dir', metavar='STRING', action='store',
#                                help='The path to the result directory',
#                                default="../results/")
#
#     groupClass = parser.add_argument_group('Classification arguments')
#     groupClass.add_argument('--CL_multiclassMethod', metavar='STRING',
#                             action='store',
#                             help='Determine which multiclass method to use if '
#                                  'the dataset is multiclass',
#                             default="oneVersusOne")
#     groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
#                             help='Determine the split ratio between learning '
#                                  'and validation sets',
#                             type=float,
#                             default=0.2)
#     groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store',
#                             help='Number of folds in cross validation',
#                             type=int, default=2)
#     groupClass.add_argument('--CL_nbClass', metavar='INT', action='store',
#                             help='Number of classes, -1 for all', type=int,
#                             default=2)
#     groupClass.add_argument('--CL_classes', metavar='STRING', action='store',
#                             nargs="+",
#                             help='Classes used in the dataset (names of the '
#                                  'folders) if not filled, random classes will '
#                                  'be '
#                                  'selected', default=["yes", "no"])
#     groupClass.add_argument('--CL_type', metavar='STRING', action='store',
#                             nargs="+",
#                             help='Determine whether to use multiview and/or '
#                                  'monoview, or Benchmark classification',
#                             default=['monoview', 'multiview'])
#     groupClass.add_argument('--CL_algos_monoview', metavar='STRING',
#                             action='store', nargs="+",
#                             help='Determine which monoview classifier to use '
#                                  'if empty, considering all',
#                             default=[''])
#     groupClass.add_argument('--CL_algos_multiview', metavar='STRING',
#                             action='store', nargs="+",
#                             help='Determine which multiview classifier to use '
#                                  'if empty, considering all',
#                             default=[''])
#     groupClass.add_argument('--CL_statsiter', metavar='INT', action='store',
#                             help="Number of iteration for each algorithm to "
#                                  "mean preds on different random states. "
#                                  "If using multiple cores, it's highly "
#                                  "recommended to use statsiter mod nbCores == "
#                                  "0",
#                             type=int,
#                             default=2)
#     groupClass.add_argument('--CL_metrics', metavar='STRING', action='store',
#                             nargs="+",
#                             help='Determine which metrics to use, separate '
#                                  'metric and configuration with ":". '
#                                  'If multiple, separate with space. If no '
#                                  'metric is specified, '
#                                  'considering all'
#                             , default=[''])
#     groupClass.add_argument('--CL_metric_princ', metavar='STRING',
#                             action='store',
#                             help='Determine which metric to use for '
#                                  'randomSearch and optimization',
#                             default="f1_score")
#     groupClass.add_argument('--CL_HPS_iter', metavar='INT', action='store',
#                             help='Determine how many hyper parameters '
#                                  'optimization tests to do',
#                             type=int, default=2)
#     groupClass.add_argument('--CL_HPS_type', metavar='STRING', action='store',
#                             help='Determine which hyperparamter search '
#                                  'function use',
#                             default="randomizedSearch")
#
#     groupRF = parser.add_argument_group('Random Forest arguments')
#     groupRF.add_argument('--RF_trees', metavar='INT', type=int, action='store',
#                          help='Number max trees',nargs="+",
#                          default=[25])
#     groupRF.add_argument('--RF_max_depth', metavar='INT', type=int,
#                          action='store',nargs="+",
#                          help='Max depth for the trees',
#                          default=[5])
#     groupRF.add_argument('--RF_criterion', metavar='STRING', action='store',
#                          help='Criterion for the trees',nargs="+",
#                          default=["entropy"])
#
#     groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
#     groupSVMLinear.add_argument('--SVML_C', metavar='INT', type=int,
#                                 action='store', nargs="+", help='Penalty parameter used',
#                                 default=[1])
#
#     groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
#     groupSVMRBF.add_argument('--SVMRBF_C', metavar='INT', type=int,
#                              action='store', nargs="+", help='Penalty parameter used',
#                              default=[1])
#
#     groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
#     groupSVMPoly.add_argument('--SVMPoly_C', metavar='INT', type=int,
#                               action='store', nargs="+", help='Penalty parameter used',
#                               default=[1])
#     groupSVMPoly.add_argument('--SVMPoly_deg', nargs="+", metavar='INT', type=int,
#                               action='store', help='Degree parameter used',
#                               default=[2])
#
#     groupAdaboost = parser.add_argument_group('Adaboost arguments')
#     groupAdaboost.add_argument('--Ada_n_est', metavar='INT', type=int,
#                                action='store', nargs="+", help='Number of estimators',
#                                default=[2])
#     groupAdaboost.add_argument('--Ada_b_est', metavar='STRING', action='store',
#                                help='Estimators',nargs="+",
#                                default=['DecisionTreeClassifier'])
#
#     groupAdaboostPregen = parser.add_argument_group('AdaboostPregen arguments')
#     groupAdaboostPregen.add_argument('--AdP_n_est', metavar='INT', type=int,
#                                      action='store',nargs="+",
#                                      help='Number of estimators',
#                                      default=[100])
#     groupAdaboostPregen.add_argument('--AdP_b_est', metavar='STRING',
#                                      action='store',nargs="+",
#                                      help='Estimators',
#                                      default=['DecisionTreeClassifier'])
#     groupAdaboostPregen.add_argument('--AdP_stumps', metavar='INT', type=int,
#                                      action='store',nargs="+",
#                                      help='Number of stumps inthe '
#                                           'pregenerated dataset',
#                                      default=[1])
#
#     groupAdaboostGraalpy = parser.add_argument_group(
#         'AdaboostGraalpy arguments')
#     groupAdaboostGraalpy.add_argument('--AdG_n_iter', metavar='INT', type=int,
#                                       action='store',nargs="+",
#                                       help='Number of estimators',
#                                       default=[100])
#     groupAdaboostGraalpy.add_argument('--AdG_stumps', metavar='INT', type=int,
#                                       action='store',nargs="+",
#                                       help='Number of stumps inthe '
#                                            'pregenerated dataset',
#                                       default=[1])
#
#     groupDT = parser.add_argument_group('Decision Trees arguments')
#     groupDT.add_argument('--DT_depth', metavar='INT', type=int, action='store',
#                          help='Determine max depth for Decision Trees',nargs="+",
#                          default=[3])
#     groupDT.add_argument('--DT_criterion', metavar='STRING', action='store',
#                          help='Determine max depth for Decision Trees',nargs="+",
#                          default=["entropy"])
#     groupDT.add_argument('--DT_splitter', metavar='STRING', action='store',
#                          help='Determine criterion for Decision Trees',nargs="+",
#                          default=["random"])
#
#     groupDTP = parser.add_argument_group('Decision Trees pregen arguments')
#     groupDTP.add_argument('--DTP_depth', metavar='INT', type=int,
#                           action='store',nargs="+",
#                           help='Determine max depth for Decision Trees',
#                           default=[3])
#     groupDTP.add_argument('--DTP_criterion', metavar='STRING', action='store',
#                           help='Determine max depth for Decision Trees',nargs="+",
#                           default=["entropy"])
#     groupDTP.add_argument('--DTP_splitter', metavar='STRING', action='store',
#                           help='Determine criterion for Decision Trees',nargs="+",
#                           default=["random"])
#     groupDTP.add_argument('--DTP_stumps', metavar='INT', type=int,
#                           action='store',nargs="+",
#                           help='Determine the number of stumps for Decision '
#                                'Trees pregen',
#                           default=[1])
#
#     groupSGD = parser.add_argument_group('SGD arguments')
#     groupSGD.add_argument('--SGD_alpha', metavar='FLOAT', type=float,
#                           action='store',nargs="+",
#                           help='Determine alpha for SGDClassifier', default=[0.1])
#     groupSGD.add_argument('--SGD_loss', metavar='STRING', action='store',
#                           help='Determine loss for SGDClassifier',nargs="+",
#                           default=['log'])
#     groupSGD.add_argument('--SGD_penalty', metavar='STRING', action='store',
#                           help='Determine penalty for SGDClassifier', nargs="+",
#                           default=['l2'])
#
#     groupKNN = parser.add_argument_group('KNN arguments')
#     groupKNN.add_argument('--KNN_neigh', metavar='INT', type=int,
#                           action='store',nargs="+",
#                           help='Determine number of neighbors for KNN',
#                           default=[1])
#     groupKNN.add_argument('--KNN_weights', nargs="+",
#                           metavar='STRING', action='store',
#                           help='Determine number of neighbors for KNN',
#                           default=["distance"])
#     groupKNN.add_argument('--KNN_algo', metavar='STRING', action='store',
#                           help='Determine number of neighbors for KNN',
#                           default=["auto"],nargs="+", )
#     groupKNN.add_argument('--KNN_p', metavar='INT', nargs="+",
#                           type=int, action='store',
#                           help='Determine number of neighbors for KNN',
#                           default=[1])
#
#     groupSCM = parser.add_argument_group('SCM arguments')
#     groupSCM.add_argument('--SCM_max_rules', metavar='INT', type=int,
#                           action='store', nargs="+",
#                           help='Max number of rules for SCM', default=[1])
#     groupSCM.add_argument('--SCM_p', metavar='FLOAT', type=float,
#                           action='store', nargs="+",
#                           help='Max number of rules for SCM', default=[1.0])
#     groupSCM.add_argument('--SCM_model_type', metavar='STRING', action='store',
#                           help='Max number of rules for SCM', nargs="+",
#                           default=["conjunction"])
#
#     groupSCMPregen = parser.add_argument_group('SCMPregen arguments')
#     groupSCMPregen.add_argument('--SCP_max_rules', metavar='INT', type=int,
#                                 action='store',nargs="+",
#                                 help='Max number of rules for SCM', default=[1])
#     groupSCMPregen.add_argument('--SCP_p', metavar='FLOAT', type=float,
#                                 action='store',nargs="+",
#                                 help='Max number of rules for SCM', default=[1.0])
#     groupSCMPregen.add_argument('--SCP_model_type', metavar='STRING',
#                                 action='store',nargs="+",
#                                 help='Max number of rules for SCM',
#                                 default=["conjunction"])
#     groupSCMPregen.add_argument('--SCP_stumps', metavar='INT', type=int,
#                                 action='store',nargs="+",
#                                 help='Number of stumps per attribute',
#                                 default=[1])
#
#     groupSCMSparsity = parser.add_argument_group('SCMSparsity arguments')
#     groupSCMSparsity.add_argument('--SCS_max_rules', metavar='INT', type=int,
#                                   action='store',nargs="+",
#                                   help='Max number of rules for SCM', default=[1])
#     groupSCMSparsity.add_argument('--SCS_stumps', metavar='INT', type=int,
#                                   action='store',nargs="+",
#                                   help='Number of stumps', default=[1])
#     groupSCMSparsity.add_argument('--SCS_p', metavar='FLOAT', type=float,
#                                   action='store',nargs="+",
#                                   help='Max number of rules for SCM',
#                                   default=[1.0])
#     groupSCMSparsity.add_argument('--SCS_model_type', metavar='STRING',
#                                   action='store',nargs="+",
#                                   help='Max number of rules for SCM',
#                                   default=["conjunction"])
#
#     groupCQBoost = parser.add_argument_group('CQBoost arguments')
#     groupCQBoost.add_argument('--CQB_mu', metavar='FLOAT', type=float,
#                               action='store',nargs="+",
#                               help='Set the mu parameter for CQBoost',
#                               default=[0.001])
#     groupCQBoost.add_argument('--CQB_epsilon', metavar='FLOAT', type=float,
#                               action='store',nargs="+",
#                               help='Set the epsilon parameter for CQBoost',
#                               default=[1e-06])
#     groupCQBoost.add_argument('--CQB_stumps', metavar='INT', type=int,
#                               action='store',nargs="+",
#                               help='Set the number of stumps for CQBoost',
#                               default=[1])
#     groupCQBoost.add_argument('--CQB_n_iter', metavar='INT', type=int,
#                               action='store',nargs="+",
#                               help='Set the maximum number of iteration in '
#                                    'CQBoost',
#                               default=[None])
#
#     groupCQBoostv2 = parser.add_argument_group('CQBoostv2 arguments')
#     groupCQBoostv2.add_argument('--CQB2_mu', metavar='FLOAT', type=float,
#                                 action='store',nargs="+",
#                                 help='Set the mu parameter for CQBoostv2',
#                                 default=[0.002])
#     groupCQBoostv2.add_argument('--CQB2_epsilon', metavar='FLOAT', type=float,
#                                 action='store',nargs="+",
#                                 help='Set the epsilon parameter for CQBoostv2',
#                                 default=[1e-08])
#
#     groupCQBoostv21 = parser.add_argument_group('CQBoostv21 arguments')
#     groupCQBoostv21.add_argument('--CQB21_mu', metavar='FLOAT', type=float,
#                                  action='store',nargs="+",
#                                  help='Set the mu parameter for CQBoostv2',
#                                  default=[0.001])
#     groupCQBoostv21.add_argument('--CQB21_epsilon', metavar='FLOAT', type=float,
#                                  action='store',nargs="+",
#                                  help='Set the epsilon parameter for CQBoostv2',
#                                  default=[1e-08])
#
#     groupQarBoost = parser.add_argument_group('QarBoost arguments')
#     groupQarBoost.add_argument('--QarB_mu', metavar='FLOAT', type=float,
#                                action='store',nargs="+",
#                                help='Set the mu parameter for QarBoost',
#                                default=[0.001])
#     groupQarBoost.add_argument('--QarB_epsilon', metavar='FLOAT', type=float,
#                                action='store',nargs="+",
#                                help='Set the epsilon parameter for QarBoost',
#                                default=[1e-08])
#
#     groupCGreed = parser.add_argument_group('CGreed arguments')
#     groupCGreed.add_argument('--CGR_stumps', metavar='INT', type=int,
#                              action='store',nargs="+",
#                              help='Set the n_stumps_per_attribute parameter '
#                                   'for CGreed',
#                              default=[1])
#     groupCGreed.add_argument('--CGR_n_iter', metavar='INT', type=int,
#                              action='store',nargs="+",
#                              help='Set the n_max_iterations parameter for '
#                                   'CGreed',
#                              default=[100])
#
#     groupCGDesc = parser.add_argument_group('CGDesc arguments')
#     groupCGDesc.add_argument('--CGD_stumps', nargs="+",  metavar='INT', type=int,
#                              action='store',
#                              help='Set the n_stumps_per_attribute parameter '
#                                   'for CGreed',
#                              default=[1])
#     groupCGDesc.add_argument('--CGD_n_iter', metavar='INT', type=int,
#                              action='store', nargs="+",
#                              help='Set the n_max_iterations parameter for '
#                                   'CGreed',
#                              default=[10])
#
#     groupCBBoost= parser.add_argument_group('CBBoost arguments')
#     groupCBBoost.add_argument('--CBB_stumps', nargs="+", metavar='INT', type=int,
#                              action='store',
#                              help='Set the n_stumps_per_attribute parameter '
#                                   'for CBBoost',
#                              default=[1])
#     groupCBBoost.add_argument('--CBB_n_iter', metavar='INT', type=int,
#                              action='store', nargs="+",
#                              help='Set the n_max_iterations parameter for '
#                                   'CBBoost',
#                              default=[100])
#
#     groupCGDescTree = parser.add_argument_group('CGDesc arguments')
#     groupCGDescTree.add_argument('--CGDT_trees', metavar='INT', type=int,
#                                  action='store', nargs="+",
#                                  help='Set thenumber of trees for CGreed',
#                                  default=[100])
#     groupCGDescTree.add_argument('--CGDT_n_iter', metavar='INT', type=int,
#                                  action='store', nargs="+",
#                                  help='Set the n_max_iterations parameter for '
#                                       'CGreed',
#                                  default=[100])
#     groupCGDescTree.add_argument('--CGDT_max_depth', metavar='INT', type=int,
#                                  action='store', nargs="+",
#                                  help='Set the n_max_iterations parameter for CGreed',
#                                  default=[2])
#
#     groupMinCQGraalpyTree = parser.add_argument_group(
#         'MinCQGraalpyTree arguments')
#     groupMinCQGraalpyTree.add_argument('--MCGT_mu', metavar='FLOAT', type=float,
#                                        action='store', nargs="+",
#                                        help='Set the mu_parameter for MinCQGraalpy',
#                                        default=[0.05])
#     groupMinCQGraalpyTree.add_argument('--MCGT_trees', metavar='INT', type=int,
#                                        action='store', nargs="+",
#                                        help='Set the n trees parameter for MinCQGraalpy',
#                                        default=[100])
#     groupMinCQGraalpyTree.add_argument('--MCGT_max_depth', metavar='INT',
#                                        type=int,nargs="+",
#                                        action='store',
#                                        help='Set the n_stumps_per_attribute parameter for MinCQGraalpy',
#                                        default=[2])
#
#     groupCQBoostTree = parser.add_argument_group('CQBoostTree arguments')
#     groupCQBoostTree.add_argument('--CQBT_mu', metavar='FLOAT', type=float,
#                                   action='store',nargs="+",
#                                   help='Set the mu parameter for CQBoost',
#                                   default=[0.001])
#     groupCQBoostTree.add_argument('--CQBT_epsilon', metavar='FLOAT', type=float,
#                                   action='store',nargs="+",
#                                   help='Set the epsilon parameter for CQBoost',
#                                   default=[1e-06])
#     groupCQBoostTree.add_argument('--CQBT_trees', metavar='INT', type=int,
#                                   action='store',nargs="+",
#                                   help='Set the number of trees for CQBoost',
#                                   default=[100])
#     groupCQBoostTree.add_argument('--CQBT_max_depth', metavar='INT', type=int,
#                                   action='store',nargs="+",
#                                   help='Set the number of stumps for CQBoost',
#                                   default=[2])
#     groupCQBoostTree.add_argument('--CQBT_n_iter', metavar='INT', type=int,
#                                   action='store',nargs="+",
#                                   help='Set the maximum number of iteration in CQBoostTree',
#                                   default=[None])
#
#     groupSCMPregenTree = parser.add_argument_group('SCMPregenTree arguments')
#     groupSCMPregenTree.add_argument('--SCPT_max_rules', metavar='INT', type=int,
#                                     action='store',nargs="+",
#                                     help='Max number of rules for SCM',
#                                     default=[1])
#     groupSCMPregenTree.add_argument('--SCPT_p', metavar='FLOAT', type=float,
#                                     action='store',nargs="+",
#                                     help='Max number of rules for SCM',
#                                     default=[1.0])
#     groupSCMPregenTree.add_argument('--SCPT_model_type', metavar='STRING',
#                                     action='store',nargs="+",
#                                     help='Max number of rules for SCM',
#                                     default=["conjunction"])
#     groupSCMPregenTree.add_argument('--SCPT_trees', metavar='INT', type=int,
#                                     action='store',nargs="+",
#                                     help='Number of stumps per attribute',
#                                     default=[100])
#     groupSCMPregenTree.add_argument('--SCPT_max_depth', metavar='INT', type=int,
#                                     action='store',nargs="+",
#                                     help='Max_depth of the trees',
#                                     default=[1])
#
#     groupSCMSparsityTree = parser.add_argument_group(
#         'SCMSparsityTree arguments')
#     groupSCMSparsityTree.add_argument('--SCST_max_rules', metavar='INT',
#                                       type=int,nargs="+",
#                                       action='store',
#                                       help='Max number of rules for SCM',
#                                       default=[1])
#     groupSCMSparsityTree.add_argument('--SCST_p', metavar='FLOAT', type=float,
#                                       action='store',nargs="+",
#                                       help='Max number of rules for SCM',
#                                       default=[1.0])
#     groupSCMSparsityTree.add_argument('--SCST_model_type', metavar='STRING',
#                                       action='store',nargs="+",
#                                       help='Max number of rules for SCM',
#                                       default=["conjunction"])
#     groupSCMSparsityTree.add_argument('--SCST_trees', metavar='INT', type=int,
#                                       action='store',nargs="+",
#                                       help='Number of stumps per attribute',
#                                       default=[100])
#     groupSCMSparsityTree.add_argument('--SCST_max_depth', metavar='INT',
#                                       type=int,nargs="+",
#                                       action='store',
#                                       help='Max_depth of the trees',
#                                       default=[1])
#
#     groupAdaboostPregenTree = parser.add_argument_group(
#         'AdaboostPregenTrees arguments')
#     groupAdaboostPregenTree.add_argument('--AdPT_n_est', metavar='INT',
#                                          type=int,nargs="+",
#                                          action='store',
#                                          help='Number of estimators',
#                                          default=[100])
#     groupAdaboostPregenTree.add_argument('--AdPT_b_est', metavar='STRING',
#                                          action='store',nargs="+",
#                                          help='Estimators',
#                                          default=['DecisionTreeClassifier'])
#     groupAdaboostPregenTree.add_argument('--AdPT_trees', metavar='INT',
#                                          type=int,nargs="+",
#                                          action='store',
#                                          help='Number of trees in the pregenerated dataset',
#                                          default=[100])
#     groupAdaboostPregenTree.add_argument('--AdPT_max_depth', metavar='INT',
#                                          type=int,nargs="+",
#                                          action='store',
#                                          help='Number of stumps inthe pregenerated dataset',
#                                          default=[3])
#
#     groupLasso = parser.add_argument_group('Lasso arguments')
#     groupLasso.add_argument('--LA_n_iter', metavar='INT', type=int,
#                             action='store',nargs="+",
#                             help='Set the max_iter parameter for Lasso',
#                             default=[1])
#     groupLasso.add_argument('--LA_alpha', metavar='FLOAT', type=float,
#                             action='store',nargs="+",
#                             help='Set the alpha parameter for Lasso',
#                             default=[1.0])
#
#     groupGradientBoosting = parser.add_argument_group(
#         'Gradient Boosting arguments')
#     groupGradientBoosting.add_argument('--GB_n_est', metavar='INT', type=int,
#                                        action='store',nargs="+",
#                                        help='Set the n_estimators_parameter for Gradient Boosting',
#                                        default=[100])
#
#     groupMinCQ = parser.add_argument_group('MinCQ arguments')
#     groupMinCQ.add_argument('--MCQ_mu', metavar='FLOAT', type=float,
#                             action='store',nargs="+",
#                             help='Set the mu_parameter for MinCQ',
#                             default=[0.05])
#     groupMinCQ.add_argument('--MCQ_stumps', metavar='INT', type=int,
#                             action='store',nargs="+",
#                             help='Set the n_stumps_per_attribute parameter for MinCQ',
#                             default=[1])
#
#     groupMinCQGraalpy = parser.add_argument_group('MinCQGraalpy arguments')
#     groupMinCQGraalpy.add_argument('--MCG_mu', metavar='FLOAT', type=float,
#                                    action='store',nargs="+",
#                                    help='Set the mu_parameter for MinCQGraalpy',
#                                    default=[0.05])
#     groupMinCQGraalpy.add_argument('--MCG_stumps', metavar='INT', type=int,
#                                    action='store',nargs="+",
#                                    help='Set the n_stumps_per_attribute parameter for MinCQGraalpy',
#                                    default=[1])
#
#     groupQarBoostv3 = parser.add_argument_group('QarBoostv3 arguments')
#     groupQarBoostv3.add_argument('--QarB3_mu', metavar='FLOAT', type=float,
#                                  action='store',nargs="+",
#                                  help='Set the mu parameter for QarBoostv3',
#                                  default=[0.001])
#     groupQarBoostv3.add_argument('--QarB3_epsilon', metavar='FLOAT', type=float,
#                                  action='store',nargs="+",
#                                  help='Set the epsilon parameter for QarBoostv3',
#                                  default=[1e-08])
#
#     groupQarBoostNC = parser.add_argument_group('QarBoostNC arguments')
#     groupQarBoostNC.add_argument('--QarBNC_mu', metavar='FLOAT', type=float,
#                                  action='store',nargs="+",
#                                  help='Set the mu parameter for QarBoostNC',
#                                  default=[0.001])
#     groupQarBoostNC.add_argument('--QarBNC_epsilon', metavar='FLOAT',
#                                  type=float, action='store',nargs="+",
#                                  help='Set the epsilon parameter for QarBoostNC',
#                                  default=[1e-08])
#
#     groupQarBoostNC2 = parser.add_argument_group('QarBoostNC2 arguments')
#     groupQarBoostNC2.add_argument('--QarBNC2_mu', metavar='FLOAT', type=float,
#                                   action='store',nargs="+",
#                                   help='Set the mu parameter for QarBoostNC2',
#                                   default=[0.001])
#     groupQarBoostNC2.add_argument('--QarBNC2_epsilon', metavar='FLOAT',
#                                   type=float, action='store',nargs="+",
#                                   help='Set the epsilon parameter for QarBoostNC2',
#                                   default=[1e-08])
#
#     groupQarBoostNC3 = parser.add_argument_group('QarBoostNC3 arguments')
#     groupQarBoostNC3.add_argument('--QarBNC3_mu', metavar='FLOAT', type=float,
#                                   action='store',nargs="+",
#                                   help='Set the mu parameter for QarBoostNC3',
#                                   default=[0.001])
#     groupQarBoostNC3.add_argument('--QarBNC3_epsilon', metavar='FLOAT',
#                                   type=float, action='store',nargs="+",
#                                   help='Set the epsilon parameter for QarBoostNC3',
#                                   default=[1e-08])
#
# #
# # multiview args
# #
#
#     groupMumbo = parser.add_argument_group('Mumbo arguments')
#     groupMumbo.add_argument('--MU_types', metavar='STRING', action='store',
#                             nargs="+",
#                             help='Determine which monoview classifier to use with Mumbo',
#                             default=[''])
#     groupMumbo.add_argument('--MU_config', metavar='STRING', action='store',
#                             nargs='+',
#                             help='Configuration for the monoview classifier in Mumbo'
#                                  ' separate each classifier with sapce and each argument with:',
#                             default=[''])
#     groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
#                             help='Max number of iteration, min number of iteration, convergence threshold',
#                             type=float,
#                             default=[10, 1, 0.01])
#     groupMumbo.add_argument('--MU_combination', action='store_true',
#                             help='Try all the monoview classifiers combinations for each view',
#                             default=False)
#
#     groupFusion = parser.add_argument_group('fusion arguments')
#     groupFusion.add_argument('--FU_types', metavar='STRING', action='store',
#                              nargs="+",
#                              help='Determine which type of fusion to use',
#                              default=[''])
#     groupEarlyFusion = parser.add_argument_group('Early fusion arguments')
#     groupEarlyFusion.add_argument('--FU_early_methods', metavar='STRING',
#                                   action='store', nargs="+",
#                                   help='Determine which early fusion method of fusion to use',
#                                   default=[''])
#     groupEarlyFusion.add_argument('--FU_E_method_configs', metavar='STRING',
#                                   action='store', nargs='+',
#                                   help='Configuration for the early fusion methods separate '
#                                        'method by space and values by :',
#                                   default=[''])
#     groupEarlyFusion.add_argument('--FU_E_cl_config', metavar='STRING',
#                                   action='store', nargs='+',
#                                   help='Configuration for the monoview classifiers '
#                                        ' used separate classifier by space '
#                                        'and configs must be of form argument1_name:value,'
#                                        'argument2_name:value',
#                                   default=[''])
#     groupEarlyFusion.add_argument('--FU_E_cl_names', metavar='STRING',
#                                   action='store', nargs='+',
#                                   help='Name of the classifiers used for each early fusion method',
#                                   default=[''])
#
#     groupLateFusion = parser.add_argument_group('Late fusion arguments')
#     groupLateFusion.add_argument('--FU_late_methods', metavar='STRING',
#                                  action='store', nargs="+",
#                                  help='Determine which late fusion method of fusion to use',
#                                  default=[''])
#     groupLateFusion.add_argument('--FU_L_method_config', metavar='STRING',
#                                  action='store', nargs='+',
#                                  help='Configuration for the fusion method',
#                                  default=[''])
#     groupLateFusion.add_argument('--FU_L_cl_config', metavar='STRING',
#                                  action='store', nargs='+',
#                                  help='Configuration for the monoview classifiers used',
#                                  default=[''])
#     groupLateFusion.add_argument('--FU_L_cl_names', metavar='STRING',
#                                  action='store', nargs="+",
#                                  help='Names of the classifier used for late fusion',
#                                  default=[''])
#     groupLateFusion.add_argument('--FU_L_select_monoview', metavar='STRING',
#                                  action='store',
#                                  help='Determine which method to use to select the monoview classifiers',
#                                  default="intersect")
#
#     groupFatLateFusion = parser.add_argument_group('Fat Late fusion arguments')
#     groupFatLateFusion.add_argument('--FLF_weights', metavar='FLOAT',
#                                     action='store', nargs="+",
#                                     help='Determine the weights of each monoview decision for FLF',
#                                     type=float,
#                                     default=[])
#
#     groupFatSCMLateFusion = parser.add_argument_group(
#         'Fat SCM Late fusion arguments')
#     groupFatSCMLateFusion.add_argument('--FSCMLF_p', metavar='FLOAT',
#                                        action='store',
#                                        help='Determine the p argument of the SCM',
#                                        type=float,
#                                        default=0.5)
#     groupFatSCMLateFusion.add_argument('--FSCMLF_max_attributes', metavar='INT',
#                                        action='store',
#                                        help='Determine the maximum number of aibutes used by the SCM',
#                                        type=int,
#                                        default=4)
#     groupFatSCMLateFusion.add_argument('--FSCMLF_model', metavar='STRING',
#                                        action='store',
#                                        help='Determine the model type of the SCM',
#                                        default="conjunction")
#
#     groupDisagreeFusion = parser.add_argument_group(
#         'Disagreement based fusion arguments')
#     groupDisagreeFusion.add_argument('--DGF_weights', metavar='FLOAT',
#                                      action='store', nargs="+",
#                                      help='Determine the weights of each monoview decision for DFG',
#                                      type=float,
#                                      default=[])

    args = parser.parse_args(arguments)
    return args


def init_random_state(random_state_arg, directory):
    r"""
    Used to init a random state.
    If no random state is specified, it will generate a 'random' seed.
    If the `randomSateArg` is a string containing only numbers, it will be converted in
     an int to generate a seed.
    If the `randomSateArg` is a string with letters, it must be a path to a pickled random
    state file that will be loaded.
    The function will also pickle the new random state in a file tobe able to retrieve it later.
    Tested


    Parameters
    ----------
    random_state_arg : None or string
        See function description.
    directory : string
        Path to the results directory.

    Returns
    -------
    random_state : numpy.random.RandomState object
        This random state will be used all along the benchmark .
    """
    if random_state_arg is None:
        random_state = np.random.RandomState(random_state_arg)
    else:
        try:
            seed = int(random_state_arg)
            random_state = np.random.RandomState(seed)
        except ValueError:
            file_name = random_state_arg
            with open(file_name, 'rb') as handle:
                random_state = pickle.load(handle)
    with open(directory + "randomState.pickle", "wb") as handle:
        pickle.dump(random_state, handle)
    return random_state


def init_stats_iter_random_states(stats_iter, random_state):
    r"""
    Used to initialize multiple random states if needed because of multiple statistical iteration of the same benchmark

    Parameters
    ----------
    stats_iter : int
        Number of statistical iterations of the same benchmark done (with a different random state).
    random_state : numpy.random.RandomState object
        The random state of the whole experimentation, that will be used to generate the ones for each
        statistical iteration.

    Returns
    -------
    stats_iter_random_states : list of numpy.random.RandomState objects
        Multiple random states, one for each sattistical iteration of the same benchmark.
    """
    if stats_iter > 1:
        stats_iter_random_states = [
            np.random.RandomState(random_state.randint(5000)) for _ in
            range(stats_iter)]
    else:
        stats_iter_random_states = [random_state]
    return stats_iter_random_states


def get_database_function(name, type_var):
    r"""Used to get the right database extraction function according to the type of database and it's name

    Parameters
    ----------
    name : string
        Name of the database.
    type_var : string
        type of dataset hdf5 or csv

    Returns
    -------
    getDatabase : function
        The function that will be used to extract the database
    """
    if name not in ["Fake", "Plausible"]:
        get_database = getattr(DB, "getClassicDB" + type_var[1:])
    else:
        get_database = getattr(DB, "get" + name + "DB" + type_var[1:])
    return get_database


def init_log_file(name, views, cl_type, log, debug, label, result_directory, add_noise, noise_std):
    r"""Used to init the directory where the preds will be stored and the log file.

    First this function will check if the result directory already exists (only one per minute is allowed).

    If the the result directory name is available, it is created, and the logfile is initiated.

    Parameters
    ----------
    name : string
        Name of the database.
    views : list of strings
        List of the view names that will be used in the benchmark.
    cl_type : list of strings
        Type of benchmark that will be made .
    log : bool
        Whether to show the log file in console or hide it.
    debug : bool
        for debug option
    label : str  for label

    result_directory : str name of the result directory

    add_noise : bool for add noise

    noise_std : level of std noise

    Returns
    -------
    results_directory : string
        Reference to the main results directory for the benchmark.
    """
    noise_string = "/n_"+str(int(noise_std*100))
    if debug:
        result_directory = result_directory + name + noise_string + \
                           "/debug_started_" + \
                           time.strftime(
                               "%Y_%m_%d-%H_%M_%S") + "_" + label + "/"
    else:
        result_directory = result_directory + name + noise_string+ "/started_" + time.strftime(
            "%Y_%m_%d-%H_%M") + "_" + label + "/"
    log_file_name = time.strftime("%Y_%m_%d-%H_%M") + "-" + ''.join(
        cl_type) + "-" + "_".join(
        views) + "-" + name + "-LOG"
    if os.path.exists(os.path.dirname(result_directory)):
        raise NameError("The result dir already exists, wait 1 min and retry")
    os.makedirs(os.path.dirname(result_directory + log_file_name))
    log_file = result_directory + log_file_name
    log_file += ".log"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        filename=log_file, level=logging.DEBUG,
                        filemode='w')
    if log:
        logging.getLogger().addHandler(logging.StreamHandler())

    return result_directory


def gen_splits(labels, split_ratio, stats_iter_random_states):
    r"""Used to _gen the train/test splits using one or multiple random states.

    Parameters
    ----------
    labels : numpy.ndarray
        Name of the database.
    split_ratio : float
        The ratio of examples between train and test set.
    stats_iter_random_states : list of numpy.random.RandomState
        The random states for each statistical iteration.

    Returns
    -------
    splits : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.
    """
    indices = np.arange(len(labels))
    splits = []
    for random_state in stats_iter_random_states:
        folds_obj = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                                   random_state=random_state,
                                                                   test_size=split_ratio)
        folds = folds_obj.split(indices, labels)
        for fold in folds:
            train_fold, test_fold = fold
        train_indices = indices[train_fold]
        test_indices = indices[test_fold]
        splits.append([train_indices, test_indices])

    return splits


def gen_k_folds(stats_iter, nb_folds, stats_iter_random_states):
    r"""Used to generate folds indices for cross validation for each statistical iteration.

    Parameters
    ----------
    stats_iter : integer
        Number of statistical iterations of the benchmark.
    nb_folds : integer
        The number of cross-validation folds for the benchmark.
    stats_iter_random_states : list of numpy.random.RandomState
        The random states for each statistical iteration.

    Returns
    -------
    folds_list : list of list of sklearn.model_selection.StratifiedKFold
        For each statistical iteration a Kfold stratified (keeping the ratio between classes in each fold).
    """
    if stats_iter > 1:
        folds_list = []
        for random_state in stats_iter_random_states:
            folds_list.append(
                sklearn.model_selection.StratifiedKFold(n_splits=nb_folds,
                                                        random_state=random_state))
    else:
        folds_list = [sklearn.model_selection.StratifiedKFold(n_splits=nb_folds,
                                                             random_state=stats_iter_random_states)]
    return folds_list


def init_views(dataset, arg_views):
    r"""Used to return the views names that will be used by the
    benchmark, their indices and all the views names.

    Parameters
    ----------
    datset : HDF5 dataset file
        The full dataset that wil be used by the benchmark.
    arg_views : list of strings
        The views that will be used by the benchmark (arg).

    Returns
    -------
    views : list of strings
        Names of the views that will be used by the benchmark.
    view_indices : list of ints
        The list of the indices of the view that will be used in the benchmark (according to the dataset).
    all_views : list of strings
        Names of all the available views in the dataset.
    """
    nb_view = dataset.get("Metadata").attrs["nbView"]
    if arg_views != ["all"]:
        allowed_views = arg_views
        all_views = [str(dataset.get("View" + str(view_index)).attrs["name"])
                    if type(
            dataset.get("View" + str(view_index)).attrs["name"]) != bytes
                    else dataset.get("View" + str(view_index)).attrs[
            "name"].decode("utf-8")
                    for view_index in range(nb_view)]
        views = []
        views_indices = []
        for view_index in range(nb_view):
            view_name = dataset.get("View" + str(view_index)).attrs["name"]
            if type(view_name) == bytes:
                view_name = view_name.decode("utf-8")
            if view_name in allowed_views:
                views.append(view_name)
                views_indices.append(view_index)
    else:
        views = [str(dataset.get("View" + str(viewIndex)).attrs["name"])
                 if type(
            dataset.get("View" + str(viewIndex)).attrs["name"]) != bytes
                 else dataset.get("View" + str(viewIndex)).attrs["name"].decode(
            "utf-8")
                 for viewIndex in range(nb_view)]
        views_indices = range(nb_view)
        all_views = views
    return views, views_indices, all_views


def gen_direcorties_names(directory, statsIter):
    r"""Used to generate the different directories of each iteration if needed.

    Parameters
    ----------
    directory : string
        Path to the results directory.
    statsIter : int
        The number of statistical iterations.

    Returns
    -------
    directories : list of strings
        Paths to each statistical iterations result directory.
    """
    if statsIter > 1:
        directories = []
        for i in range(statsIter):
            directories.append(directory + "iter_" + str(i + 1) + "/")
    else:
        directories = [directory]
    return directories


def find_dataset_names(path, type, names):
    """This function goal is to browse the dataset directory and extarcts all
     the needed dataset names."""
    available_file_names = [file_name.strip().split(".")[0]
                            for file_name in os.listdir(path)
                            if file_name.endswith(type)]
    if names == ["all"]:
        return available_file_names
    elif len(names)>1:
        return [used_name for used_name in available_file_names if used_name in names]
    else:
        return names


def gen_argument_dictionaries(labels_dictionary, directories, multiclass_labels,
                              labels_combinations, indices_multiclass,
                              hyper_param_search, args, k_folds,
                              stats_iter_random_states, metrics,
                              argument_dictionaries,
                              benchmark, nb_views, views, views_indices):
    r"""Used to generate a dictionary for each benchmark.

    One for each label combination (if multiclass), for each statistical iteration, generates an dictionary with
    all necessary information to perform the benchmark

    Parameters
    ----------
    labels_dictionary : dictionary
        Dictionary mapping labels indices to labels names.
    directories : list of strings
        List of the paths to the result directories for each statistical iteration.
    multiclass_labels : list of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.
    labels_combinations : list of lists of numpy.ndarray
        Each original couple of different labels.
    indices_multiclass : list of lists of numpy.ndarray
        For each combination, contains a biclass labels numpy.ndarray with the 0/1 labels of combination.
    hyper_param_search : string
        Type of hyper parameter optimization method
    args : parsed args objects
        All the args passed by the user.
    k_folds : list of list of sklearn.model_selection.StratifiedKFold
        For each statistical iteration a Kfold stratified (keeping the ratio between classes in each fold).
    stats_iter_random_states : list of numpy.random.RandomState objects
        Multiple random states, one for each sattistical iteration of the same benchmark.
    metrics : list of lists
        metrics that will be used to evaluate the algorithms performance.
    argument_dictionaries : dictionary
        Dictionary resuming all the specific arguments for the benchmark, oe dictionary for each classifier.
    benchmark : dictionary
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    nb_views : int
        THe number of views used by the benchmark.
    views : list of strings
        List of the names of the used views.
    views_indices : list of ints
        List of indices (according to the dataset) of the used views.

    Returns
    -------
    benchmarkArgumentDictionaries : list of dicts
        All the needed arguments for the benchmarks.

    """
    benchmark_argument_dictionaries = []
    for combination_index, labels_combination in enumerate(labels_combinations):
        for iter_index, iterRandomState in enumerate(stats_iter_random_states):
            benchmark_argument_dictionary = {
                "LABELS_DICTIONARY": {0: labels_dictionary[labels_combination[0]],
                                      1: labels_dictionary[
                                          labels_combination[1]]},
                "directory": directories[iter_index] +
                             labels_dictionary[labels_combination[0]] +
                             "-vs-" +
                             labels_dictionary[labels_combination[1]] + "/",
                "classificationIndices": [
                    indices_multiclass[combination_index][0][iter_index],
                    indices_multiclass[combination_index][1][iter_index],
                    indices_multiclass[combination_index][2][iter_index]],
                "args": args,
                "labels": multiclass_labels[combination_index],
                "kFolds": k_folds[iter_index],
                "randomState": iterRandomState,
                "hyperParamSearch": hyper_param_search,
                "metrics": metrics,
                "argumentDictionaries": argument_dictionaries,
                "benchmark": benchmark,
                "views": views,
                "viewsIndices": views_indices,
                "flag": [iter_index, labels_combination]}
            benchmark_argument_dictionaries.append(benchmark_argument_dictionary)
    return benchmark_argument_dictionaries
