# Import built-in modules
import argparse
import pkgutil  # for TimeStamp in CSVFile
import os
import time
import sys
import select
import logging
import errno
import cPickle

# Import 3rd party modules
from joblib import Parallel, delayed
import numpy as np
import math
import matplotlib
import sklearn

# Import own modules
import Multiview
import Metrics
import MonoviewClassifiers
from Multiview.ExecMultiview import ExecMultiview, ExecMultiview_multicore
from Monoview.ExecClassifMonoView import ExecMonoview, ExecMonoview_multicore
import Multiview.GetMultiviewDb as DB
from Versions import testVersions
from ResultAnalysis import resultAnalysis, analyzeLabels, analyzeIterResults

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

matplotlib.use('Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure


def initLogFile(args):
    resultDirectory = "../../Results/" + args.name + "/started_" + time.strftime("%Y_%m_%d-%H_%M") + "/"
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


def input_(timeout=15):
    print "You have " + str(timeout) + " seconds to stop the script by typing n"

    i, o, e = select.select([sys.stdin], [], [], timeout)

    if i:
        return sys.stdin.readline().strip()
    else:
        return "y"


def confirm(resp=True, timeout=15):
    ans = input_(timeout)
    if not ans:
        return resp
    if ans not in ['y', 'Y', 'n', 'N']:
        print 'please enter y or n.'
    if ans == 'y' or ans == 'Y':
        return True
    if ans == 'n' or ans == 'N':
        return False


def initMultipleDatasets(args, nbCores):
    """Used to create copies of the dataset if multicore computation is used
    Needs arg.pathF and arg.name"""
    if nbCores > 1:
        if DB.datasetsAlreadyExist(args.pathF, args.name, nbCores):
            logging.debug("Info:\t Enough copies of the dataset are already available")
            pass
        else:
            logging.debug("Start:\t Creating " + str(nbCores) + " temporary datasets for multiprocessing")
            logging.warning(" WARNING : /!\ This may use a lot of HDD storage space : " +
                            str(os.path.getsize(args.pathF + args.name + ".hdf5") * nbCores / float(
                                1024) / 1000 / 1000) + " Gbytes /!\ ")
            confirmation = confirm()
            if not confirmation:
                sys.exit(0)
            else:
                datasetFiles = DB.copyHDF5(args.pathF, args.name, nbCores)
                logging.debug("Start:\t Creating datasets for multiprocessing")
                return datasetFiles


def initViews(DATASET, args):
    """Used to return the views names that will be used by the algos, their indices and all the views names
    Needs args.views"""
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


def initBenchmark(args):
    """Used to create a list of all the algorithm packages names used for the benchmark
    Needs args.CL_type, args.CL_algos_multiview, args.MU_types, args.FU_types, args.FU_late_methods,
    args.FU_early_methods, args.CL_algos_monoview"""
    benchmark = {"Monoview": {}, "Multiview": {}}
    allMultiviewPackages = [name for _, name, isPackage
                            in pkgutil.iter_modules(['Multiview/']) if isPackage]
    if args.CL_type == ["Benchmark"]:

        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['MonoviewClassifiers'])
                            if (not isPackage)]
        benchmark["Monoview"] = allMonoviewAlgos
        benchmark["Multiview"] = dict((multiviewPackageName, "_") for multiviewPackageName in allMultiviewPackages)
        for multiviewPackageName in allMultiviewPackages:
            multiviewPackage = getattr(Multiview, multiviewPackageName)
            multiviewModule = getattr(multiviewPackage, multiviewPackageName)
            benchmark = multiviewModule.getBenchmark(benchmark, args=args)

    if "Multiview" in args.CL_type:
        benchmark["Multiview"] = {}
        if args.CL_algos_multiview == [""]:
            algosMutliview = allMultiviewPackages
        else:
            algosMutliview = args.CL_algos_multiview
        for multiviewPackageName in allMultiviewPackages:
            if multiviewPackageName in algosMutliview:
                multiviewPackage = getattr(Multiview, multiviewPackageName)
                multiviewModule = getattr(multiviewPackage, multiviewPackageName)
                benchmark = multiviewModule.getBenchmark(benchmark, args=args)
    if "Monoview" in args.CL_type:
        if args.CL_algos_monoview == ['']:
            benchmark["Monoview"] = [name for _, name, isPackage in pkgutil.iter_modules(["MonoviewClassifiers"])
                                     if not isPackage]

        else:
            benchmark["Monoview"] = args.CL_algos_monoview
    return benchmark


def initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, DATASET, NB_CLASS, kwargsInit):
    if benchmark["Monoview"]:
        argumentDictionaries["Monoview"] = []
        for view in views:
            for classifier in benchmark["Monoview"]:
                if classifier == "SCM":
                    if DATASET.get("View" + str(allViews.index(view))).attrs["binary"]:
                        arguments = {
                            "args": {classifier + "KWARGS": kwargsInit[classifier + "KWARGSInit"], "feat": view,
                                     "CL_type": classifier, "nbClass": NB_CLASS}, "viewIndex": allViews.index(view)}
                        argumentDictionaries["Monoview"].append(arguments)
                    else:
                        pass
                else:
                    arguments = {
                        "args": {classifier + "KWARGS": kwargsInit[classifier + "KWARGSInit"], "feat": view,
                                 "CL_type": classifier, "nbClass": NB_CLASS}, "viewIndex": allViews.index(view)}
                    argumentDictionaries["Monoview"].append(arguments)
    return argumentDictionaries


def initMonoviewKWARGS(args, classifiersNames):
    monoviewKWARGS = {}
    for classifiersName in classifiersNames:
        classifierModule = getattr(MonoviewClassifiers, classifiersName)
        monoviewKWARGS[classifiersName + "KWARGSInit"] = classifierModule.getKWARGS(
            [(key, value) for key, value in vars(args).iteritems() if key.startswith("CL_" + classifiersName)])
    return monoviewKWARGS


def initKWARGS(args, benchmark):
    if "Monoview" in benchmark:
        monoviewKWARGS = initMonoviewKWARGS(args, benchmark["Monoview"])
    else:
        monoviewKWARGS = {}
    return monoviewKWARGS


def lateFusionSetArgs(views, viewsIndices, classes, method,
                      classifiersNames, classifiersConfig, fusionMethodConfig):
    arguments = {"CL_type": "Fusion",
                 "views": views,
                 "NB_VIEW": len(views),
                 "viewsIndices": viewsIndices,
                 "NB_CLASS": len(classes),
                 "LABELS_NAMES": args.CL_classes,
                 "FusionKWARGS": {"fusionType": "LateFusion", "fusionMethod": method,
                                  "classifiersNames": classifiersNames,
                                  "classifiersConfigs": classifiersConfig,
                                  'fusionMethodConfig': fusionMethodConfig,
                                  "nbView": (len(viewsIndices))}}
    return arguments


def initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries, randomState, directory, resultsMonoview, classificationIndices):
    multiviewArguments = []
    if "Multiview" in benchmark:
        for multiviewAlgoName in benchmark["Multiview"]:
            multiviewPackage = getattr(Multiview, multiviewAlgoName)
            mutliviewModule = getattr(multiviewPackage, multiviewAlgoName)
            multiviewArguments += mutliviewModule.getArgs(args, benchmark, views, viewsIndices, randomState, directory, resultsMonoview, classificationIndices)
    argumentDictionaries["Multiview"] = multiviewArguments
    return argumentDictionaries


def arangeMetrics(metrics, metricPrinc):
    if [metricPrinc] in metrics:
        metricIndex = metrics.index([metricPrinc])
        firstMetric = metrics[0]
        metrics[0] = [metricPrinc]
        metrics[metricIndex] = firstMetric
    else:
        raise AttributeError(metricPrinc + " not in metric pool")
    return metrics


def genSplits(statsIter, indices, DATASET, splitRatio, statsIterRandomStates):
    if statsIter > 1:
        splits = []
        for randomState in statsIterRandomStates:
            trainIndices, testIndices, a, b = sklearn.model_selection.train_test_split(indices, DATASET.get("Labels").value,
                                                                                       test_size=splitRatio,
                                                                                       random_state=randomState)
            splits.append([trainIndices, testIndices])
        return splits
    else:
        trainIndices, testIndices, a, b = sklearn.model_selection.train_test_split(indices, DATASET.get("Labels").value,
                                                                                   test_size=splitRatio,
                                                                                   random_state=statsIterRandomStates)
        return trainIndices, testIndices


def genKFolds(statsIter, nbFolds, statsIterRandomStates):
    if statsIter > 1:
        foldsList = []
        for randomState in statsIterRandomStates:
            foldsList.append(sklearn.model_selection.KFold(n_splits=nbFolds, random_state=randomState))
        return foldsList
    else:
        return sklearn.model_selection.KFold(n_splits=nbFolds, random_state=statsIterRandomStates)


def genDirecortiesNames(directory, statsIter):
    if statsIter>1:
        directories = []
        for i in range(statsIter):
            directories.append(directory+"iter_"+str(i+1)+"/")
        return directories
    else:
        return directory


def classifyOneIter_multicore(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices, kFolds,
                              randomState, hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime, start, benchmark,
                              views):
    resultsMonoview = []
    labelsNames = LABELS_DICTIONARY.values()
    resultsMonoview += [ExecMonoview_multicore(directory, args.name, labelsNames, classificationIndices, kFolds,
                                                 coreIndex, args.type, args.pathF, randomState,
                                                 hyperParamSearch=hyperParamSearch,
                                                 metrics=metrics, nIter=args.CL_GS_iter,
                                                 **arguments)
                         for arguments in argumentDictionaries["Monoview"]]
    monoviewTime = time.time() - dataBaseTime - start

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries, randomState, directory, resultsMonoview, classificationIndices)

    resultsMultiview = []
    resultsMultiview += [
        ExecMultiview_multicore(directory, coreIndex, args.name, classificationIndices, kFolds, args.type,
                                args.pathF, LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
                                metrics=metrics, nIter=args.CL_GS_iter,**arguments)
        for arguments in argumentDictionaries["Multiview"]]
    multiviewTime = time.time() - monoviewTime - dataBaseTime - start

    labels = np.array(
        [resultMonoview[1][3] for resultMonoview in resultsMonoview] + [resultMultiview[3] for resultMultiview in
                                                                        resultsMultiview]).transpose()
    trueLabels = DATASET.get("Labels").value
    times = [dataBaseTime, monoviewTime, multiviewTime]
    results = (resultsMonoview, resultsMultiview)
    analyzeLabels(labels, trueLabels, results, directory)
    logging.debug("Start:\t Analyze Global Results for iteration")
    resultAnalysis(benchmark, results, args.name, times, metrics, directory)
    logging.debug("Done:\t Analyze Global Results for iteration")
    globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
    totalTime = time.time() - start
    logging.info("Extraction time : "+str(dataBaseTime)+
                 "s, Monoview time : "+str(monoviewTime)+
                 "s, Multiview Time : "+str(multiviewTime)+
                 "s, Global Analysis Time : "+str(globalAnalysisTime)+
                 "s, Total Duration : "+str(totalTime)+"s")
    return results


def classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices, kFolds,
                    randomState, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
                    benchmark, views):
    resultsMonoview = []
    labelsNames = LABELS_DICTIONARY.values()
    if nbCores > 1:
        nbExperiments = len(argumentDictionaries["Monoview"])
        for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
            resultsMonoview += (Parallel(n_jobs=nbCores)(
                delayed(ExecMonoview_multicore)(directory, args.name, labelsNames, classificationIndices, kFolds,
                                                coreIndex, args.type, args.pathF, randomState,
                                                hyperParamSearch=hyperParamSearch,
                                                metrics=metrics, nIter=args.CL_GS_iter,
                                                **argumentDictionaries["Monoview"][coreIndex + stepIndex * nbCores])
                for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))

    else:
        resultsMonoview += ([ExecMonoview(directory, DATASET.get("View" + str(arguments["viewIndex"])),
                                          DATASET.get("Labels").value, args.name, labelsNames,
                                          classificationIndices, kFolds, 1, args.type, args.pathF, randomState,
                                          hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=args.CL_GS_iter,
                                          **arguments)
                             for arguments in argumentDictionaries["Monoview"]])
    monoviewTime = time.time() - dataBaseTime - start

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries, randomState, directory, resultsMonoview, classificationIndices)

    resultsMultiview = []
    if nbCores > 1:
        nbExperiments = len(argumentDictionaries["Multiview"])
        for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
            resultsMultiview += Parallel(n_jobs=nbCores)(
                delayed(ExecMultiview_multicore)(directory, coreIndex, args.name, classificationIndices, kFolds, args.type,
                                                 args.pathF,
                                                 LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
                                                 metrics=metrics, nIter=args.CL_GS_iter,
                                                 **argumentDictionaries["Multiview"][stepIndex * nbCores + coreIndex])
                for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores)))
    else:
        resultsMultiview = [
            ExecMultiview(directory, DATASET, args.name, classificationIndices, kFolds, 1, args.type, args.pathF,
                          LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
                          metrics=metrics, nIter=args.CL_GS_iter, **arguments) for arguments in
            argumentDictionaries["Multiview"]]
    multiviewTime = time.time() - monoviewTime - dataBaseTime - start
    if nbCores > 1:
        logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
        datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
        logging.debug("Start:\t Deleting datasets for multiprocessing")
    labels = np.array(
        [resultMonoview[1][3] for resultMonoview in resultsMonoview] + [resultMultiview[3] for resultMultiview in
                                                                        resultsMultiview]).transpose()
    trueLabels = DATASET.get("Labels").value
    times = [dataBaseTime, monoviewTime, multiviewTime]
    results = (resultsMonoview, resultsMultiview)
    analyzeLabels(labels, trueLabels, results, directory)
    logging.debug("Start:\t Analyze Global Results")
    resultAnalysis(benchmark, results, args.name, times, metrics, directory)
    logging.debug("Done:\t Analyze Global Results")
    globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
    totalTime = time.time() - start
    logging.info("Extraction time : "+str(dataBaseTime)+
                 "s, Monoview time : "+str(monoviewTime)+
                 "s, Multiview Time : "+str(multiviewTime)+
                 "s, Global Analysis Time : "+str(globalAnalysisTime)+
                 "s, Total Duration : "+str(totalTime)+"s")
    return results


def initRandomState(randomStateArg, directory):
    if randomStateArg is None:
        randomState = np.random.RandomState(randomStateArg)
    else:
        try:
            seed = int(randomStateArg)
            randomState = np.random.RandomState(seed)
        except ValueError:
            fileName = randomStateArg
            with open(fileName, 'rb') as handle:
                randomState = cPickle.load(handle)
    with open(directory+"randomState.pickle", "wb") as handle:
        cPickle.dump(randomState, handle)
    return randomState


testVersions()
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
                        default=0.3)
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
                        help="Number of iteration for each algorithm to mean results if using multiple cores, it's highly recommended to use statsiter mod(nbCores) = 0", type=int,
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
                        default=['DecisionTree', 'DecisionTree', 'DecisionTree'])
groupMumbo.add_argument('--MU_config', metavar='STRING', action='store', nargs='+',
                        help='Configuration for the monoview classifier in Mumbo',
                        default=['2:0.5', '2:0.5', '2:0.5'])
groupMumbo.add_argument('--MU_iter', metavar='INT', action='store', nargs=3,
                        help='Max number of iteration, min number of iteration, convergence threshold', type=float,
                        default=[10, 1, 0.01])

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

start = time.time()
args = parser.parse_args()

os.nice(args.nice)
nbCores = args.CL_cores
statsIter = args.CL_statsiter
hyperParamSearch = args.CL_HPS_type

directory = initLogFile(args)
randomState = initRandomState(args.randomState, directory)
if statsIter > 1:
    statsIterRandomStates = [np.random.RandomState(randomState.randint(500)) for _ in range(statsIter)]
else:
    statsIterRandomStates = randomState

if args.name not in ["MultiOmic", "ModifiedMultiOmic", "Caltech", "Fake", "Plausible", "KMultiOmic"]:
    getDatabase = getattr(DB, "getClassicDB" + args.type[1:])
else:
    getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])


DATASET, LABELS_DICTIONARY = getDatabase(args.views, args.pathF, args.name, args.CL_nb_class,
                                         args.CL_classes)

datasetLength = DATASET.get("Metadata").attrs["datasetLength"]
indices = np.arange(datasetLength)
classificationIndices = genSplits(statsIter, indices, DATASET, args.CL_split, statsIterRandomStates)

kFolds = genKFolds(statsIter, args.CL_nbFolds, statsIterRandomStates)

datasetFiles = initMultipleDatasets(args, nbCores)

views, viewsIndices, allViews = initViews(DATASET, args)
if not views:
    raise ValueError, "Empty views list, modify selected views to match dataset " + args.views

NB_VIEW = len(views)
NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]

metrics = [metric.split(":") for metric in args.CL_metrics]
if metrics == [[""]]:
    metricsNames = [name for _, name, isPackage
                    in pkgutil.iter_modules(['Metrics']) if not isPackage and name != "log_loss"]
    metrics = [[metricName] for metricName in metricsNames]
    metrics = arangeMetrics(metrics, args.CL_metric_princ)
for metricIndex, metric in enumerate(metrics):
    if len(metric) == 1:
        metrics[metricIndex] = [metric[0], None]

logging.info("Start:\t Finding all available mono- & multiview algorithms")

benchmark = initBenchmark(args)


initKWARGS = initKWARGS(args, benchmark)

dataBaseTime = time.time() - start

argumentDictionaries = {"Monoview": [], "Multiview": []}
argumentDictionaries = initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, DATASET, NB_CLASS,
                                             initKWARGS)
directories = genDirecortiesNames(directory, statsIter)

if statsIter>1:
    for statIterIndex in range(statsIter):
        if not os.path.exists(os.path.dirname(directories[statIterIndex]+"train_labels.csv")):
            try:
                os.makedirs(os.path.dirname(directories[statIterIndex]+"train_labels.csv"))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        trainIndices, testIndices = classificationIndices[statIterIndex]
        trainLabels = DATASET.get("Labels").value[trainIndices]
        np.savetxt(directories[statIterIndex]+"train_labels.csv", trainLabels, delimiter=",")
    if nbCores > 1:
        iterResults = []
        nbExperiments = statsIter
        for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
            iterResults += (Parallel(n_jobs=nbCores)(
                delayed(classifyOneIter_multicore)(LABELS_DICTIONARY, argumentDictionaries, 1, directories[coreIndex + stepIndex * nbCores], args, classificationIndices[coreIndex + stepIndex * nbCores], kFolds[coreIndex + stepIndex * nbCores],
                                                   statsIterRandomStates[coreIndex + stepIndex * nbCores], hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime, start, benchmark,
                                                   views)
                for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))
        logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
        datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
        logging.debug("Start:\t Deleting datasets for multiprocessing")
    else:
        iterResults = []
        for iterIndex in range(statsIter):
            iterResults.append(classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directories[iterIndex], args,
                                               classificationIndices[iterIndex], kFolds[iterIndex], statsIterRandomStates[iterIndex],
                                               hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start, benchmark, views))
    analyzeIterResults(iterResults, args.name, metrics, directory)

else:
    res = classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directories, args, classificationIndices, kFolds,
                    statsIterRandomStates, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
                    benchmark, views)

if statsIter > 1:
    pass