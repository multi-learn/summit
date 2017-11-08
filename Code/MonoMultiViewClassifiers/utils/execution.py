import argparse
import numpy as np
import pickle
import time
import os
import errno
import logging
import sklearn


def parseTheArgs(arguments):
    """Used to parse the args entered by the user"""

    parser = argparse.ArgumentParser(
        description='This file is used to benchmark the scores fo multiple classification algorithm on multiview data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('-log', action='store_true', help='Use option to activate logging to console')
    groupStandard.add_argument('--name', metavar='STRING', action='store', help='Name of Database (default: %(default)s)',
                               default='Plausible')
    groupStandard.add_argument('--type', metavar='STRING', action='store',
                               help='Type of database : .hdf5 or .csv (default: %(default)s)',
                               default='.hdf5')
    groupStandard.add_argument('--views', metavar='STRING', action='store', nargs="+",
                               help='Name of the views selected for learning (default: %(default)s)',
                               default=[''])
    groupStandard.add_argument('--pathF', metavar='STRING', action='store', help='Path to the hdf5 dataset or database '
                                                                                 'folder (default: %(default)s)',
                               default='../Data/')
    groupStandard.add_argument('--nice', metavar='INT', action='store', type=int,
                               help='Niceness for the processes', default=0)
    groupStandard.add_argument('--randomState', metavar='STRING', action='store',
                               help="The random state seed to use or the path to a pickle file where it is stored",
                               default=None)
    groupStandard.add_argument('--nbCores', metavar='INT', action='store', help='Number of cores to use for parallel '
                                                                                'computing, -1 for all',
                               type=int, default=2)
    groupStandard.add_argument('--machine', metavar='STRING', action='store',
                               help='Type of machine on which the script runs', default="PC")

    groupClass = parser.add_argument_group('Classification arguments')
    groupClass.add_argument('--CL_multiclassMethod', metavar='STRING', action='store',
                            help='Determine which multiclass method to use if the dataset is multiclass',
                            default="oneVersusOne")
    groupClass.add_argument('--CL_split', metavar='FLOAT', action='store',
                            help='Determine the split ratio between learning and validation sets', type=float,
                            default=0.2)
    groupClass.add_argument('--CL_nbFolds', metavar='INT', action='store', help='Number of folds in cross validation',
                            type=int, default=2)
    groupClass.add_argument('--CL_nbClass', metavar='INT', action='store', help='Number of classes, -1 for all', type=int,
                            default=2)
    groupClass.add_argument('--CL_classes', metavar='STRING', action='store', nargs="+",
                            help='Classes used in the dataset (names of the folders) if not filled, random classes will be '
                                 'selected', default=["yes", "no"])
    groupClass.add_argument('--CL_type', metavar='STRING', action='store', nargs="+",
                            help='Determine whether to use Multiview and/or Monoview, or Benchmark classification',
                            default=['Benchmark'])
    groupClass.add_argument('--CL_algos_monoview', metavar='STRING', action='store', nargs="+",
                            help='Determine which monoview classifier to use if empty, considering all',
                            default=[''])
    groupClass.add_argument('--CL_algos_multiview', metavar='STRING', action='store', nargs="+",
                            help='Determine which multiview classifier to use if empty, considering all',
                            default=[''])
    groupClass.add_argument('--CL_statsiter', metavar='INT', action='store',
                            help="Number of iteration for each algorithm to mean preds on different random states. "
                                 "If using multiple cores, it's highly recommended to use statsiter mod nbCores == 0",
                            type=int,
                            default=2)
    groupClass.add_argument('--CL_metrics', metavar='STRING', action='store', nargs="+",
                            help='Determine which metrics to use, separate metric and configuration with ":".'
                                 ' If multiple, separate with space. If no metric is specified, '
                                 'considering all'
                            , default=[''])
    groupClass.add_argument('--CL_metric_princ', metavar='STRING', action='store',
                            help='Determine which metric to use for randomSearch and optimization', default="f1_score")
    groupClass.add_argument('--CL_GS_iter', metavar='INT', action='store',
                            help='Determine how many hyper parameters optimization tests to do', type=int, default=2)
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
    args = parser.parse_args(arguments)
    return args


def initRandomState(randomStateArg, directory):
    """Used to init a random state and multiple if needed (multicore)"""
    if randomStateArg is None:
        randomState = np.random.RandomState(randomStateArg)
    else:
        try:
            seed = int(randomStateArg)
            randomState = np.random.RandomState(seed)
        except ValueError:
            fileName = randomStateArg
            with open(fileName, 'rb') as handle:
                randomState = pickle.load(handle)
    with open(directory + "randomState.pickle", "wb") as handle:
        pickle.dump(randomState, handle)
    return randomState


def initLogFile(args):
    """Used to init the directory where the preds will be stored and the log file"""
    resultDirectory = "../Results/" + args.name + "/started_" + time.strftime("%Y_%m_%d-%H_%M") + "/"
    logFileName = time.strftime("%Y%m%d-%H%M%S") + "-" + ''.join(args.CL_type) + "-" + "_".join(
        args.views) + "-" + args.name + "-LOG"
    if not os.path.exists(os.path.dirname(resultDirectory + logFileName)):
        try:
            os.makedirs(os.path.dirname(resultDirectory + logFileName))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    logFile = resultDirectory + logFileName
    if os.path.isfile(logFile + ".log"):
        for i in range(1, 20):
            testFileName = logFileName + "-" + str(i) + ".log"
            if not (os.path.isfile(resultDirectory + testFileName)):
                logFile = resultDirectory + testFileName
                break
    else:
        logFile += ".log"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
                        filemode='w')
    if args.log:
        logging.getLogger().addHandler(logging.StreamHandler())

    return resultDirectory


def genSplits(labels, splitRatio, statsIterRandomStates):
    """Used to gen the train/test splits using one or multiple random states"""
    indices = np.arange(len(labels))
    splits = []
    for randomState in statsIterRandomStates:
        foldsObj = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                                  random_state=randomState,
                                                                  test_size=splitRatio)
        folds = foldsObj.split(indices, labels)
        for fold in folds:
            train_fold, test_fold = fold
        trainIndices = indices[train_fold]
        testIndices = indices[test_fold]
        splits.append([trainIndices, testIndices])

    return splits


def genKFolds(statsIter, nbFolds, statsIterRandomStates):
    """Used to generate folds indices for cross validation and multiple if needed"""
    if statsIter > 1:
        foldsList = []
        for randomState in statsIterRandomStates:
            foldsList.append(sklearn.model_selection.StratifiedKFold(n_splits=nbFolds, random_state=randomState))
        return foldsList
    else:
        return sklearn.model_selection.StratifiedKFold(n_splits=nbFolds, random_state=statsIterRandomStates)


def initViews(DATASET, args):
    """Used to return the views names that will be used by the algos, their indices and all the views names"""
    NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
    if args.views != [""]:
        allowedViews = args.views
        allViews = [str(DATASET.get("View" + str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
        views = [str(DATASET.get("View" + str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW) if
                 str(DATASET.get("View" + str(viewIndex)).attrs["name"]) in allowedViews]
        viewsIndices = [viewIndex for viewIndex in range(NB_VIEW) if
                        str(DATASET.get("View" + str(viewIndex)).attrs["name"]) in allowedViews]
        return views, viewsIndices, allViews
    else:
        views = [str(DATASET.get("View" + str(viewIndex)).attrs["name"]) for viewIndex in range(NB_VIEW)]
        viewsIndices = np.arange(NB_VIEW)
        allViews = views
        return views, viewsIndices, allViews


def genDirecortiesNames(directory, statsIter):
    """Used to generate the different directories of each iteration if needed"""
    if statsIter > 1:
        directories = []
        for i in range(statsIter):
            directories.append(directory + "iter_" + str(i + 1) + "/")
    else:
        directories = [directory]
    return directories


def genArgumentDictionaries(labelsDictionary, directories, multiclassLabels, labelsCombinations, oldIndicesMulticlass, hyperParamSearch, args,
                            kFolds, statsIterRandomStates, metrics, argumentDictionaries, benchmark, nbViews, views):
    benchmarkArgumentDictionaries = []
    for combinationIndex, labelsCombination in enumerate(labelsCombinations):
        for iterIndex, iterRandomState in enumerate(statsIterRandomStates):
            benchmarkArgumentDictionary = {"LABELS_DICTIONARY": {0:labelsDictionary[labelsCombination[0]],
                                                                 1:labelsDictionary[labelsCombination[1]]},
                                           "directory": directories[iterIndex]+
                                                        labelsDictionary[labelsCombination[0]]+
                                                        "vs"+
                                                        labelsDictionary[labelsCombination[1]]+"/",
                                           "classificationIndices": oldIndicesMulticlass[combinationIndex][iterIndex],
                                           "args": args,
                                           "labels": multiclassLabels[combinationIndex],
                                           "kFolds": kFolds[iterIndex],
                                           "randomState": iterRandomState,
                                           "hyperParamSearch": hyperParamSearch,
                                           "metrics": metrics,
                                           "argumentDictionaries": argumentDictionaries,
                                           "benchmark": benchmark,
                                           "views": views,
                                           "viewsIndices": range(nbViews),
                                           "flag": [iterIndex, labelsCombination]}
            benchmarkArgumentDictionaries.append(benchmarkArgumentDictionary)
    return benchmarkArgumentDictionaries


