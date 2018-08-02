import argparse
import numpy as np
import pickle
import time
import os
import errno
import logging
import sklearn


from . import GetMultiviewDb as DB

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
    groupStandard.add_argument('-full', action='store_true', help='Use option to use full dataset and no labels or view filtering')
    groupStandard.add_argument('-debug', action='store_true',
                               help='Use option to bebug implemented algorithms')


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
    groupClass.add_argument('--CL_HPS_iter', metavar='INT', action='store',
                            help='Determine how many hyper parameters optimization tests to do', type=int, default=2)
    groupClass.add_argument('--CL_HPS_type', metavar='STRING', action='store',
                            help='Determine which hyperparamter search function use', default="randomizedSearch")

    groupRF = parser.add_argument_group('Random Forest arguments')
    groupRF.add_argument('--RF_trees', metavar='INT', type=int, action='store', help='Number max trees',
                         default=25)
    groupRF.add_argument('--RF_max_depth', metavar='INT', type=int, action='store',
                         help='Max depth for the trees',
                         default=5)
    groupRF.add_argument('--RF_criterion', metavar='STRING', action='store', help='Criterion for the trees',
                         default="entropy")

    groupSVMLinear = parser.add_argument_group('Linear SVM arguments')
    groupSVMLinear.add_argument('--SVML_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                                default=1)

    groupSVMRBF = parser.add_argument_group('SVW-RBF arguments')
    groupSVMRBF.add_argument('--SVMRBF_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                             default=1)

    groupSVMPoly = parser.add_argument_group('Poly SVM arguments')
    groupSVMPoly.add_argument('--SVMPoly_C', metavar='INT', type=int, action='store', help='Penalty parameter used',
                              default=1)
    groupSVMPoly.add_argument('--SVMPoly_deg', metavar='INT', type=int, action='store', help='Degree parameter used',
                              default=2)

    groupAdaboost = parser.add_argument_group('Adaboost arguments')
    groupAdaboost.add_argument('--Ada_n_est', metavar='INT', type=int, action='store', help='Number of estimators',
                               default=2)
    groupAdaboost.add_argument('--Ada_b_est', metavar='STRING', action='store', help='Estimators',
                               default='DecisionTreeClassifier')

    groupDT = parser.add_argument_group('Decision Trees arguments')
    groupDT.add_argument('--DT_depth', metavar='INT', type=int, action='store',
                         help='Determine max depth for Decision Trees', default=3)
    groupDT.add_argument('--DT_criterion', metavar='STRING', action='store',
                         help='Determine max depth for Decision Trees', default="entropy")
    groupDT.add_argument('--DT_splitter', metavar='STRING', action='store',
                         help='Determine criterion for Decision Trees', default="random")

    groupSGD = parser.add_argument_group('SGD arguments')
    groupSGD.add_argument('--SGD_alpha', metavar='FLOAT', type=float, action='store',
                          help='Determine alpha for SGDClassifier', default=0.1)
    groupSGD.add_argument('--SGD_loss', metavar='STRING', action='store',
                          help='Determine loss for SGDClassifier', default='log')
    groupSGD.add_argument('--SGD_penalty', metavar='STRING', action='store',
                          help='Determine penalty for SGDClassifier', default='l2')

    groupKNN = parser.add_argument_group('KNN arguments')
    groupKNN.add_argument('--KNN_neigh', metavar='INT', type=int, action='store',
                          help='Determine number of neighbors for KNN', default=1)
    groupKNN.add_argument('--KNN_weights', metavar='STRING', action='store',
                          help='Determine number of neighbors for KNN', default="distance")
    groupKNN.add_argument('--KNN_algo', metavar='STRING', action='store',
                          help='Determine number of neighbors for KNN', default="auto")
    groupKNN.add_argument('--KNN_p', metavar='INT', type=int, action='store',
                          help='Determine number of neighbors for KNN', default=1)

    groupSCM = parser.add_argument_group('SCM arguments')
    groupSCM.add_argument('--SCM_max_rules', metavar='INT', type=int, action='store',
                          help='Max number of rules for SCM', default=1)
    groupSCM.add_argument('--SCM_p', metavar='FLOAT', type=float, action='store',
                          help='Max number of rules for SCM', default=1.0)
    groupSCM.add_argument('--SCM_model_type', metavar='STRING', action='store',
                          help='Max number of rules for SCM', default="conjunction")

    groupCQBoost = parser.add_argument_group('CQBoost arguments')
    groupCQBoost.add_argument('--CQB_mu', metavar='FLOAT', type=float, action='store',
                              help='Set the mu parameter for CQBoost', default=0.001)
    groupCQBoost.add_argument('--CQB_epsilon', metavar='FLOAT', type=float, action='store',
                              help='Set the epsilon parameter for CQBoost', default=1e-08)

    groupCQBoostv2 = parser.add_argument_group('CQBoostv2 arguments')
    groupCQBoostv2.add_argument('--CQB2_mu', metavar='FLOAT', type=float, action='store',
                              help='Set the mu parameter for CQBoostv2', default=0.001)
    groupCQBoostv2.add_argument('--CQB2_epsilon', metavar='FLOAT', type=float, action='store',
                              help='Set the epsilon parameter for CQBoostv2', default=1e-08)

    groupCQBoostv21 = parser.add_argument_group('CQBoostv21 arguments')
    groupCQBoostv21.add_argument('--CQB21_mu', metavar='FLOAT', type=float, action='store',
                                help='Set the mu parameter for CQBoostv2', default=0.001)
    groupCQBoostv21.add_argument('--CQB21_epsilon', metavar='FLOAT', type=float, action='store',
                                help='Set the epsilon parameter for CQBoostv2', default=1e-08)

    groupQarBoost = parser.add_argument_group('QarBoost arguments')
    groupQarBoost.add_argument('--QarB_mu', metavar='FLOAT', type=float, action='store',
                                 help='Set the mu parameter for QarBoost', default=0.001)
    groupQarBoost.add_argument('--QarB_epsilon', metavar='FLOAT', type=float, action='store',
                                 help='Set the epsilon parameter for QarBoost', default=1e-08)

    groupQarBoostv2 = parser.add_argument_group('QarBoostv2 arguments')
    groupQarBoostv2.add_argument('--QarB2_mu', metavar='FLOAT', type=float, action='store',
                               help='Set the mu parameter for QarBoostv2', default=0.001)
    groupQarBoostv2.add_argument('--QarB2_epsilon', metavar='FLOAT', type=float, action='store',
                                 help='Set the epsilon parameter for QarBoostv2', default=1e-08)

    groupQarBoostv3 = parser.add_argument_group('QarBoostv3 arguments')
    groupQarBoostv3.add_argument('--QarB3_mu', metavar='FLOAT', type=float, action='store',
                                 help='Set the mu parameter for QarBoostv3', default=0.001)
    groupQarBoostv3.add_argument('--QarB3_epsilon', metavar='FLOAT', type=float, action='store',
                                 help='Set the epsilon parameter for QarBoostv3', default=1e-08)

    groupQarBoostNC = parser.add_argument_group('QarBoostNC arguments')
    groupQarBoostNC.add_argument('--QarBNC_mu', metavar='FLOAT', type=float, action='store',
                                 help='Set the mu parameter for QarBoostNC', default=0.001)
    groupQarBoostNC.add_argument('--QarBNC_epsilon', metavar='FLOAT', type=float, action='store',
                                 help='Set the epsilon parameter for QarBoostNC', default=1e-08)

    groupQarBoostNC2 = parser.add_argument_group('QarBoostNC2 arguments')
    groupQarBoostNC2.add_argument('--QarBNC2_mu', metavar='FLOAT', type=float, action='store',
                                 help='Set the mu parameter for QarBoostNC2', default=0.001)
    groupQarBoostNC2.add_argument('--QarBNC2_epsilon', metavar='FLOAT', type=float, action='store',
                                 help='Set the epsilon parameter for QarBoostNC2', default=1e-08)


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

    groupLateFusion = parser.add_argument_group('Late Fusion arguments')
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

    groupFatLateFusion = parser.add_argument_group('Fat Late Fusion arguments')
    groupFatLateFusion.add_argument('--FLF_weights', metavar='FLOAT', action='store', nargs="+",
                                 help='Determine the weights of each monoview decision for FLF', type=float,
                                 default=[])

    groupFatSCMLateFusion = parser.add_argument_group('Fat SCM Late Fusion arguments')
    groupFatSCMLateFusion.add_argument('--FSCMLF_p', metavar='FLOAT', action='store',
                                    help='Determine the p argument of the SCM', type=float,
                                    default=0.5)
    groupFatSCMLateFusion.add_argument('--FSCMLF_max_attributes', metavar='INT', action='store',
                                    help='Determine the maximum number of aibutes used by the SCM', type=int,
                                    default=4)
    groupFatSCMLateFusion.add_argument('--FSCMLF_model', metavar='STRING', action='store',
                                    help='Determine the model type of the SCM',
                                    default="conjunction")

    groupDisagreeFusion = parser.add_argument_group('Disagreement based fusion arguments')
    groupDisagreeFusion.add_argument('--DGF_weights', metavar='FLOAT', action='store', nargs="+",
                                    help='Determine the weights of each monoview decision for DFG', type=float,
                                    default=[])



    args = parser.parse_args(arguments)
    return args


def initRandomState(randomStateArg, directory):
    r"""
    Used to init a random state.
    If no random state is specified, it will generate a 'random' seed.
    If the `randomSateArg` is a string containing only numbers, it will be converted in an int to generate a seed.
    If the `randomSateArg` is a string with letters, it must be a path to a pickled random state file that will be loaded.
    The function will also pickle the new random state in a file tobe able to retrieve it later.
    Tested


    Parameters
    ----------
    randomStateArg : None or string
        See function description.
    directory : string
        Path to the results directory.

    Returns
    -------
    randomState : numpy.random.RandomState object
        This random state will be used all along the benchmark .
    """
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


def initStatsIterRandomStates(statsIter, randomState):
    r"""
    Used to initialize multiple random states if needed because of multiple statistical iteration of the same benchmark

    Parameters
    ----------
    statsIter : int
        Number of statistical iterations of the same benchmark done (with a different random state).
    randomState : numpy.random.RandomState object
        The random state of the whole experimentation, that will be used to generate the ones for each
        statistical iteration.

    Returns
    -------
    statsIterRandomStates : list of numpy.random.RandomState objects
        Multiple random states, one for each sattistical iteration of the same benchmark.
    """
    if statsIter > 1:
        statsIterRandomStates = [np.random.RandomState(randomState.randint(5000)) for _ in range(statsIter)]
    else:
        statsIterRandomStates = [randomState]
    return statsIterRandomStates


def getDatabaseFunction(name, type):
    r"""Used to get the right database extraction function according to the type of database and it's name

    Parameters
    ----------
    name : string
        Name of the database.
    type : string
        type of dataset hdf5 or csv

    Returns
    -------
    getDatabase : function
        The function that will be used to extract the database
    """
    if name not in ["Fake", "Plausible"]:
        getDatabase = getattr(DB, "getClassicDB" + type[1:])
    else:
        getDatabase = getattr(DB, "get" + name + "DB" + type[1:])
    return getDatabase


def initLogFile(name, views, CL_type, log, debug):
    r"""Used to init the directory where the preds will be stored and the log file.

    First this function will check if the result directory already exists (only one per minute is allowed).

    If the the result directory name is available, it is created, and the logfile is initiated.

    Parameters
    ----------
    name : string
        Name of the database.
    views : list of strings
        List of the view names that will be used in the benchmark.
    CL_type : list of strings
        Type of benchmark that will be made .
    log : bool
        Whether to show the log file in console or hide it.

    Returns
    -------
    resultsDirectory : string
        Reference to the main results directory for the benchmark.
    """
    if debug:
        resultDirectory = "../Results/" + name + "/debug_started_" + time.strftime("%Y_%m_%d-%H_%M_%S") + "/"
    else:
        resultDirectory = "../Results/" + name + "/started_" + time.strftime("%Y_%m_%d-%H_%M") + "/"
    logFileName = time.strftime("%Y_%m_%d-%H_%M") + "-" + ''.join(CL_type) + "-" + "_".join(
        views) + "-" + name + "-LOG"
    if os.path.exists(os.path.dirname(resultDirectory)):
        raise NameError("The result dir already exists, wait 1 min and retry")
    os.makedirs(os.path.dirname(resultDirectory + logFileName))
    logFile = resultDirectory + logFileName
    logFile += ".log"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=logFile, level=logging.DEBUG,
                        filemode='w')
    if log:
        logging.getLogger().addHandler(logging.StreamHandler())

    return resultDirectory


def genSplits(labels, splitRatio, statsIterRandomStates):
    r"""Used to gen the train/test splits using one or multiple random states.

    Parameters
    ----------
    labels : numpy.ndarray
        Name of the database.
    splitRatio : float
        The ratio of examples between train and test set.
    statsIterRandomStates : list of numpy.random.RandomState
        The random states for each statistical iteration.

    Returns
    -------
    splits : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.
    """
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
    r"""Used to generate folds indices for cross validation for each statistical iteration.

    Parameters
    ----------
    statsIter : integer
        Number of statistical iterations of the benchmark.
    nbFolds : integer
        The number of cross-validation folds for the benchmark.
    statsIterRandomStates : list of numpy.random.RandomState
        The random states for each statistical iteration.

    Returns
    -------
    foldsList : list of list of sklearn.model_selection.StratifiedKFold
        For each statistical iteration a Kfold stratified (keeping the ratio between classes in each fold).
    """
    if statsIter > 1:
        foldsList = []
        for randomState in statsIterRandomStates:
            foldsList.append(sklearn.model_selection.StratifiedKFold(n_splits=nbFolds, random_state=randomState))
    else:
        foldsList = [sklearn.model_selection.StratifiedKFold(n_splits=nbFolds, random_state=statsIterRandomStates)]
    return foldsList


def initViews(DATASET, argViews):
    r"""Used to return the views names that will be used by the benchmark, their indices and all the views names.

    Parameters
    ----------
    DATASET : HDF5 dataset file
        The full dataset that wil be used by the benchmark.
    argViews : list of strings
        The views that will be used by the benchmark (arg).

    Returns
    -------
    views : list of strings
        Names of the views that will be used by the benchmark.
    viewIndices : list of ints
        The list of the indices of the view that will be used in the benchmark (according to the dataset).
    allViews : list of strings
        Names of all the available views in the dataset.
    """
    NB_VIEW = DATASET.get("Metadata").attrs["nbView"]
    if argViews != [""]:
        allowedViews = argViews
        allViews = [str(DATASET.get("View" + str(viewIndex)).attrs["name"])
                    if type(DATASET.get("View" + str(viewIndex)).attrs["name"])!=bytes
                    else DATASET.get("View" + str(viewIndex)).attrs["name"].decode("utf-8")
                    for viewIndex in range(NB_VIEW)]
        views = []
        viewsIndices = []
        for viewIndex in range(NB_VIEW):
            viewName = DATASET.get("View" + str(viewIndex)).attrs["name"]
            if type(viewName) == bytes:
                viewName = viewName.decode("utf-8")
            if viewName in allowedViews:
                views.append(viewName)
                viewsIndices.append(viewIndex)
    else:
        views = [str(DATASET.get("View" + str(viewIndex)).attrs["name"])
                if type(DATASET.get("View" + str(viewIndex)).attrs["name"])!=bytes
                else DATASET.get("View" + str(viewIndex)).attrs["name"].decode("utf-8")
                for viewIndex in range(NB_VIEW)]
        viewsIndices = range(NB_VIEW)
        allViews = views
    return views, viewsIndices, allViews


def genDirecortiesNames(directory, statsIter):
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


def genArgumentDictionaries(labelsDictionary, directories, multiclassLabels, labelsCombinations, indicesMulticlass,
                            hyperParamSearch, args, kFolds, statsIterRandomStates, metrics, argumentDictionaries,
                            benchmark, nbViews, views, viewsIndices):
    r"""Used to generate a dictionary for each benchmark.

    One for each label combination (if multiclass), for each statistical iteration, generates an dictionary with
    all necessary information to perform the benchmark

    Parameters
    ----------
    labelsDictionary : dictionary
        Dictionary mapping labels indices to labels names.
    directories : list of strings
        List of the paths to the result directories for each statistical iteration.
    multiclassLabels : list of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.
    labelsCombinations : list of lists of numpy.ndarray
        Each original couple of different labels.
    indicesMulticlass : list of lists of numpy.ndarray
        For each combination, contains a biclass labels numpy.ndarray with the 0/1 labels of combination.
    hyperParamSearch : string
        Type of hyper parameter optimization method
    args : parsed args objects
        All the args passed by the user.
    kFolds : list of list of sklearn.model_selection.StratifiedKFold
        For each statistical iteration a Kfold stratified (keeping the ratio between classes in each fold).
    statsIterRandomStates : list of numpy.random.RandomState objects
        Multiple random states, one for each sattistical iteration of the same benchmark.
    metrics : list of lists
        Metrics that will be used to evaluate the algorithms performance.
    argumentDictionaries : dictionary
        Dictionary resuming all the specific arguments for the benchmark, oe dictionary for each classifier.
    benchmark : dictionary
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    nbViews : int
        THe number of views used by the benchmark.
    views : list of strings
        List of the names of the used views.
    viewsIndices : list of ints
        List of indices (according to the dataset) of the used views.

    Returns
    -------
    benchmarkArgumentDictionaries : list of dicts
        All the needed arguments for the benchmarks.

    """
    benchmarkArgumentDictionaries = []
    for combinationIndex, labelsCombination in enumerate(labelsCombinations):
        for iterIndex, iterRandomState in enumerate(statsIterRandomStates):
            benchmarkArgumentDictionary = {"LABELS_DICTIONARY": {0:labelsDictionary[labelsCombination[0]],
                                                                 1:labelsDictionary[labelsCombination[1]]},
                                           "directory": directories[iterIndex]+
                                                        labelsDictionary[labelsCombination[0]]+
                                                        "-vs-"+
                                                        labelsDictionary[labelsCombination[1]]+"/",
                                           "classificationIndices": [indicesMulticlass[combinationIndex][0][iterIndex],
                                                                     indicesMulticlass[combinationIndex][1][iterIndex],
                                                                     indicesMulticlass[combinationIndex][2][iterIndex]],
                                           "args": args,
                                           "labels": multiclassLabels[combinationIndex],
                                           "kFolds": kFolds[iterIndex],
                                           "randomState": iterRandomState,
                                           "hyperParamSearch": hyperParamSearch,
                                           "metrics": metrics,
                                           "argumentDictionaries": argumentDictionaries,
                                           "benchmark": benchmark,
                                           "views": views,
                                           "viewsIndices": viewsIndices,
                                           "flag": [iterIndex, labelsCombination]}
            benchmarkArgumentDictionaries.append(benchmarkArgumentDictionary)
    return benchmarkArgumentDictionaries
