import errno
import logging
import math
import os
import pkgutil
import time

import matplotlib
import numpy as np
from joblib import Parallel, delayed
import h5py

matplotlib.use('Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure

# Import own modules
from . import MonoviewClassifiers
from . import MultiviewClassifiers
from .Multiview.ExecMultiview import ExecMultiview, ExecMultiview_multicore
from .Monoview.ExecClassifMonoView import ExecMonoview, ExecMonoview_multicore
from .utils import GetMultiviewDb as DB
from .ResultAnalysis import getResults  #resultAnalysis, analyzeLabels, analyzeIterResults, analyzeIterLabels, genNamesFromRes,
from .utils import execution, Dataset, Multiclass
from . import Metrics

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def initBenchmark(CL_type,  monoviewAlgos, multiviewAlgos, args):
    r"""Used to create a list of all the algorithm packages names used for the benchmark.

    First this function will check if the benchmark need mono- or/and multiview algorithms and adds to the right
    dictionary the asked algorithms. If none is asked by the user, all will be added.

    If the keyword `"Benchmark"` is used, all mono- and multiview algorithms will be added.

    Parameters
    ----------
    CL_type : List of string
        List of types of needed benchmark
    multiviewAlgos : List of strings
        List of multiview algorithms needed for the benchmark
    monoviewAlgos : Listof strings
        List of monoview algorithms needed for the benchmark
    args : ParsedArgumentParser args
        All the input args (used to tune the algorithms)

    Returns
    -------
    benchmark : Dictionary of dictionaries
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    """
    benchmark = {"Monoview": {}, "Multiview": {}}
    allMultiviewPackages = [name for _, name, isPackage
                            in pkgutil.iter_modules(['./MonoMultiViewClassifiers/MultiviewClassifiers/']) if isPackage]

    if "Monoview" in CL_type:
        if monoviewAlgos == ['']:
            benchmark["Monoview"] = [name for _, name, isPackage in pkgutil.iter_modules(["./MonoMultiViewClassifiers/MonoviewClassifiers"])
                                     if not isPackage]

        else:
            benchmark["Monoview"] = monoviewAlgos

    if "Multiview" in CL_type:
        benchmark["Multiview"] = {}
        if multiviewAlgos == [""]:
            algosMutliview = allMultiviewPackages
        else:
            algosMutliview = multiviewAlgos
        for multiviewPackageName in allMultiviewPackages:
            if multiviewPackageName in algosMutliview:
                multiviewPackage = getattr(MultiviewClassifiers, multiviewPackageName)
                multiviewModule = getattr(multiviewPackage, multiviewPackageName+"Module")
                benchmark = multiviewModule.getBenchmark(benchmark, args=args)

    if CL_type == ["Benchmark"]:
        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['./MonoMultiViewClassifiers/MonoviewClassifiers'])
                            if (not isPackage) and name not in ["framework"]]
        benchmark["Monoview"] = allMonoviewAlgos
        benchmark["Multiview"] = dict((multiviewPackageName, "_") for multiviewPackageName in allMultiviewPackages)
        for multiviewPackageName in allMultiviewPackages:
            multiviewPackage = getattr(MultiviewClassifiers, multiviewPackageName)
            multiviewModule = getattr(multiviewPackage, multiviewPackageName+"Module")
            benchmark = multiviewModule.getBenchmark(benchmark, args=args)

    return benchmark


def genViewsDictionnary(DATASET, views):
    r"""Used to generate a dictionary mapping a view name (key) to it's index in the dataset (value).

    Parameters
    ----------
    DATASET : `h5py` dataset file
        The full dataset on which the benchmark will be done
    views : List of strings
        Names of the selected views on which the banchmark will be done

    Returns
    -------
    viewDictionary : Dictionary
        Dictionary mapping the view names totheir indexin the full dataset.
        """
    datasetsNames = DATASET.keys()
    viewsDictionary = {}
    for datasetName in datasetsNames:
        if datasetName[:4]=="View":
            viewName = DATASET.get(datasetName).attrs["name"]
            if type(viewName)==bytes:
                viewName = viewName.decode("utf-8")
            if viewName in views:
                viewsDictionary[viewName] = int(datasetName[4:])

    return viewsDictionary


def initMonoviewExps(benchmark, viewsDictionary, nbClass, kwargsInit):
    r"""Used to add each monoview exeperience args to the list of monoview experiences args.

    First this function will check if the benchmark need mono- or/and multiview algorithms and adds to the right
    dictionary the asked algorithms. If none is asked by the user, all will be added.

    If the keyword `"Benchmark"` is used, all mono- and multiview algorithms will be added.

    Parameters
    ----------
    benchmark : dictionary
        All types of monoview and multiview experiments that have to be benchmarked
    argumentDictionaries : dictionary
        Maps monoview and multiview experiments arguments.
    viewDictionary : dictionary
        Maps the view names to their index in the HDF5 dataset
    nbClass : integer
        Number of different labels in the classification

    Returns
    -------
    benchmark : Dictionary of dictionaries
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    """
    argumentDictionaries = {"Monoview":[], "Multiview":[]}
    if benchmark["Monoview"]:
        argumentDictionaries["Monoview"] = []
        for viewName, viewIndex in viewsDictionary.items():
            for classifier in benchmark["Monoview"]:
                arguments = {
                    "args": {classifier + "KWARGS": kwargsInit[classifier + "KWARGSInit"], "feat": viewName,
                             "CL_type": classifier, "nbClass": nbClass}, "viewIndex": viewIndex}
                argumentDictionaries["Monoview"].append(arguments)
    return argumentDictionaries


def initMonoviewKWARGS(args, classifiersNames):
    r"""Used to init kwargs thanks to a function in each monoview classifier package.

    Parameters
    ----------
    args : parsed args objects
        All the args passed by the user.
    classifiersNames : list of strings
        List of the benchmarks's monoview classifiers names.

    Returns
    -------
    monoviewKWARGS : Dictionary of dictionaries
        Dictionary resuming all the specific arguments for the benchmark, one dictionary for each classifier.

        For example, for Adaboost, the KWARGS will be `{"n_estimators":<value>, "base_estimator":<value>}`"""

    logging.debug("Start:\t Initializing Monoview classifiers arguments")
    monoviewKWARGS = {}
    for classifiersName in classifiersNames:
        try:
            classifierModule = getattr(MonoviewClassifiers, classifiersName)
        except AttributeError:
            raise AttributeError(classifiersName+" is not implemented in MonoviewClassifiers, "
                                                 "please specify the name of the file in MonoviewClassifiers")
        monoviewKWARGS[classifiersName + "KWARGSInit"] = classifierModule.formatCmdArgs(args)
    logging.debug("Done:\t Initializing Monoview classifiers arguments")
    return monoviewKWARGS


def initKWARGSFunc(args, benchmark):
    monoviewKWARGS = initMonoviewKWARGS(args, benchmark["Monoview"])
    return monoviewKWARGS


def initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries, randomState, directory,
                           resultsMonoview, classificationIndices):
    """Used to add each monoview exeperience args to the list of monoview experiences args"""
    logging.debug("Start:\t Initializing Multiview classifiers arguments")
    multiviewArguments = []
    if "Multiview" in benchmark:
        for multiviewAlgoName in benchmark["Multiview"]:
            multiviewPackage = getattr(MultiviewClassifiers, multiviewAlgoName)
            mutliviewModule = getattr(multiviewPackage, multiviewAlgoName+"Module")

            multiviewArguments += mutliviewModule.getArgs(args, benchmark, views, viewsIndices, randomState, directory,
                                                          resultsMonoview, classificationIndices)
    argumentDictionaries["Multiview"] = multiviewArguments
    logging.debug("Start:\t Initializing Multiview classifiers arguments")
    return argumentDictionaries


def arangeMetrics(metrics, metricPrinc):
    """Used to get the metrics list in the right order so that
    the first one is the principal metric specified in args"""
    if [metricPrinc] in metrics:
        metricIndex = metrics.index([metricPrinc])
        firstMetric = metrics[0]
        metrics[0] = [metricPrinc]
        metrics[metricIndex] = firstMetric
    else:
        raise AttributeError(metricPrinc + " not in metric pool")
    return metrics


def benchmarkInit(directory, classificationIndices, labels, LABELS_DICTIONARY, kFolds):
    logging.debug("Start:\t Benchmark initialization")
    if not os.path.exists(os.path.dirname(directory + "train_labels.csv")):
        try:
            os.makedirs(os.path.dirname(directory + "train_labels.csv"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    trainIndices = classificationIndices[0]
    trainLabels = labels[trainIndices]
    np.savetxt(directory + "train_labels.csv", trainLabels, delimiter=",")
    np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")
    resultsMonoview = []
    folds = kFolds.split(np.arange(len(trainLabels)), trainLabels)
    minFoldLen = int(len(trainLabels)/kFolds.n_splits)
    for foldIndex, (trainCVIndices, testCVIndices) in enumerate(folds):
        fileName = directory+"/folds/test_labels_fold_"+str(foldIndex)+".csv"
        if not os.path.exists(os.path.dirname(fileName)):
            try:
                os.makedirs(os.path.dirname(fileName))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.savetxt(fileName, trainLabels[testCVIndices[:minFoldLen]], delimiter=",")
    labelsNames = list(LABELS_DICTIONARY.values())
    logging.debug("Done:\t Benchmark initialization")
    return resultsMonoview, labelsNames


def execOneBenchmark(coreIndex=-1, LABELS_DICTIONARY=None, directory=None, classificationIndices=None, args=None,
                     kFolds=None, randomState=None, hyperParamSearch=None, metrics=None, argumentDictionaries=None,
                     benchmark=None, views=None, viewsIndices=None, flag=None, labels=None,
                     ExecMonoview_multicore=ExecMonoview_multicore, ExecMultiview_multicore=ExecMultiview_multicore,
                     initMultiviewArguments=initMultiviewArguments):
    """Used to run a benchmark using one core. ExecMonoview_multicore, initMultiviewArguments and
     ExecMultiview_multicore args are only used for tests"""

    resultsMonoview, labelsNames = benchmarkInit(directory, classificationIndices, labels, LABELS_DICTIONARY, kFolds)

    logging.debug("Start:\t Monoview benchmark")
    resultsMonoview += [ExecMonoview_multicore(directory, args.name, labelsNames, classificationIndices, kFolds,
                                               coreIndex, args.type, args.pathF, randomState, labels,
                                               hyperParamSearch=hyperParamSearch, metrics=metrics,
                                               nIter=args.CL_HPS_iter, **argument)
                        for argument in argumentDictionaries["Monoview"]]
    logging.debug("Done:\t Monoview benchmark")

    logging.debug("Start:\t Multiview arguments initialization")
    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)
    logging.debug("Done:\t Multiview arguments initialization")

    logging.debug("Start:\t Multiview benchmark")
    resultsMultiview = [
        ExecMultiview_multicore(directory, coreIndex, args.name, classificationIndices, kFolds, args.type,
                                args.pathF, LABELS_DICTIONARY, randomState, labels, hyperParamSearch=hyperParamSearch,
                                metrics=metrics, nIter=args.CL_HPS_iter, **arguments)
        for arguments in argumentDictionaries["Multiview"]]
    logging.debug("Done:\t Multiview benchmark")

    return [flag, resultsMonoview + resultsMultiview]


def execOneBenchmark_multicore(nbCores=-1, LABELS_DICTIONARY=None, directory=None, classificationIndices=None, args=None,
                               kFolds=None, randomState=None, hyperParamSearch=None, metrics=None, argumentDictionaries=None,
                               benchmark=None, views=None, viewsIndices=None, flag=None, labels=None,
                               ExecMonoview_multicore=ExecMonoview_multicore,
                               ExecMultiview_multicore=ExecMultiview_multicore,
                               initMultiviewArguments=initMultiviewArguments):
    """Used to run a benchmark using multiple cores. ExecMonoview_multicore, initMultiviewArguments and
     ExecMultiview_multicore args are only used for tests"""

    resultsMonoview, labelsNames = benchmarkInit(directory, classificationIndices, labels, LABELS_DICTIONARY, kFolds)

    logging.debug("Start:\t Monoview benchmark")
    nbExperiments = len(argumentDictionaries["Monoview"])
    nbMulticoreToDo = int(math.ceil(float(nbExperiments) / nbCores))
    for stepIndex in range(nbMulticoreToDo):
        resultsMonoview += (Parallel(n_jobs=nbCores)(
            delayed(ExecMonoview_multicore)(directory, args.name, labelsNames, classificationIndices, kFolds,
                                            coreIndex, args.type, args.pathF, randomState, labels,
                                            hyperParamSearch=hyperParamSearch,
                                            metrics=metrics, nIter=args.CL_HPS_iter,
                                            **argumentDictionaries["Monoview"][coreIndex + stepIndex * nbCores])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))
    logging.debug("Done:\t Monoview benchmark")

    logging.debug("Start:\t Multiview arguments initialization")
    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)
    logging.debug("Done:\t Multiview arguments initialization")

    logging.debug("Start:\t Multiview benchmark")
    resultsMultiview = []
    nbExperiments = len(argumentDictionaries["Multiview"])
    nbMulticoreToDo = int(math.ceil(float(nbExperiments) / nbCores))
    for stepIndex in range(nbMulticoreToDo):
        resultsMultiview += Parallel(n_jobs=nbCores)(
            delayed(ExecMultiview_multicore)(directory, coreIndex, args.name, classificationIndices, kFolds,
                                             args.type, args.pathF, LABELS_DICTIONARY, randomState, labels,
                                             hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=args.CL_HPS_iter,
                                             **argumentDictionaries["Multiview"][stepIndex * nbCores + coreIndex])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores)))
    logging.debug("Done:\t Multiview benchmark")

    return [flag, resultsMonoview + resultsMultiview]


def execOneBenchmarkMonoCore(DATASET=None, LABELS_DICTIONARY=None, directory=None, classificationIndices=None, args=None,
                             kFolds=None, randomState=None, hyperParamSearch=None, metrics=None, argumentDictionaries=None,
                             benchmark=None, views=None, viewsIndices=None, flag=None, labels=None,
                             ExecMonoview_multicore=ExecMonoview_multicore, ExecMultiview_multicore=ExecMultiview_multicore,
                             initMultiviewArguments=initMultiviewArguments):

    resultsMonoview, labelsNames = benchmarkInit(directory, classificationIndices, labels, LABELS_DICTIONARY, kFolds)
    logging.debug("Start:\t Monoview benchmark")
    for arguments in argumentDictionaries["Monoview"]:
        X = DATASET.get("View"+str(arguments["viewIndex"]))
        Y = labels
        resultsMonoview += [ExecMonoview(directory, X, Y, args.name, labelsNames, classificationIndices, kFolds,
                                                   1, args.type, args.pathF, randomState,
                                                   hyperParamSearch=hyperParamSearch, metrics=metrics,
                                                   nIter=args.CL_HPS_iter, **arguments)]
    logging.debug("Done:\t Monoview benchmark")

    logging.debug("Start:\t Multiview arguments initialization")

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)
    logging.debug("Done:\t Multiview arguments initialization")

    logging.debug("Start:\t Multiview benchmark")
    resultsMultiview = []
    for arguments in argumentDictionaries["Multiview"]:
        resultsMultiview += [
            ExecMultiview(directory, DATASET, args.name, classificationIndices, kFolds, 1, args.type,
                                    args.pathF, LABELS_DICTIONARY, randomState, labels, hyperParamSearch=hyperParamSearch,
                                    metrics=metrics, nIter=args.CL_HPS_iter, **arguments)]
    logging.debug("Done:\t Multiview benchmark")

    return [flag, resultsMonoview + resultsMultiview]


def execBenchmark(nbCores, statsIter, nbMulticlass, benchmarkArgumentsDictionaries, classificationIndices, directories,
                  directory, multiClassLabels, metrics, labelsDictionary, nbLabels, DATASET,
                  execOneBenchmark=execOneBenchmark, execOneBenchmark_multicore=execOneBenchmark_multicore,
                  execOneBenchmarkMonoCore=execOneBenchmarkMonoCore, getResults=getResults, delete=DB.deleteHDF5):
    r"""Used to execute the needed benchmark(s) on multicore or mono-core functions.

    Parameters
    ----------
    nbCores : int
        Number of threads that the benchmarks can use.
    statsIter : int
        Number of statistical iterations that have to be done.
    benchmarkArgumentsDictionaries : list of dictionaries
        All the needed arguments for the benchmarks.
    classificationIndices : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.
    directories : list of strings
        List of the paths to the result directories for each statistical iteration.
    directory : string
        Path to the main results directory.
    multiClassLabels : ist of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.
    metrics : list of lists
        Metrics that will be used to evaluate the algorithms performance.
    labelsDictionary : dictionary
        Dictionary mapping labels indices to labels names.
    nbLabels : int
        Total number of different labels in the dataset.
    DATASET : HDF5 dataset file
        The full dataset that wil be used by the benchmark.
    classifiersNames : list of strings
        List of the benchmarks's monoview classifiers names.
    rest_of_the_args :
        Just used for testing purposes


    Returns
    -------
    results : list of lists
        The results of the benchmark.
    """
    logging.debug("Start:\t Executing all the needed biclass benchmarks")
    results = []
    if nbCores > 1:
        if statsIter > 1 or nbMulticlass > 1:
            nbExpsToDo = len(benchmarkArgumentsDictionaries)
            nbMulticoreToDo = range(int(math.ceil(float(nbExpsToDo) / nbCores)))
            for stepIndex in nbMulticoreToDo:
                results += (Parallel(n_jobs=nbCores)(delayed(execOneBenchmark)
                                                     (coreIndex=coreIndex,
                                                      **benchmarkArgumentsDictionaries[coreIndex + stepIndex * nbCores])
                                                     for coreIndex in range(min(nbCores, nbExpsToDo - stepIndex * nbCores))))
        else:
            results += [execOneBenchmark_multicore(nbCores=nbCores, **benchmarkArgumentsDictionaries[0])]
    else:
        for arguments in benchmarkArgumentsDictionaries:
            results += [execOneBenchmarkMonoCore(DATASET=DATASET, **arguments)]
    logging.debug("Done:\t Executing all the needed biclass benchmarks")
    if nbCores > 1:
        logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
        args = benchmarkArgumentsDictionaries[0]["args"]
        datasetFiles = delete(args.pathF, args.name, nbCores)
        logging.debug("Start:\t Deleting datasets for multiprocessing")
    # Do everything with flagging
    nbExamples = len(classificationIndices[0][0])+len(classificationIndices[0][1])
    multiclassGroundTruth = DATASET.get("Labels").value
    logging.debug("Start:\t Analyzing predictions")
    getResults(results, statsIter, nbMulticlass, benchmarkArgumentsDictionaries, multiclassGroundTruth, metrics, classificationIndices, directories, directory, labelsDictionary, nbExamples, nbLabels)
    logging.debug("Done:\t Analyzing predictions")

    return results


def execClassif(arguments):
    """Main function to execute the benchmark"""
    start = time.time()
    args = execution.parseTheArgs(arguments)

    os.nice(args.nice)
    nbCores = args.nbCores
    if nbCores == 1:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    statsIter = args.CL_statsiter
    hyperParamSearch = args.CL_HPS_type
    multiclassMethod = args.CL_multiclassMethod
    CL_type = args.CL_type
    monoviewAlgos = args.CL_algos_monoview
    multiviewAlgos = args.CL_algos_multiview

    directory = execution.initLogFile(args.name, args.views, args.CL_type, args.log, args.debug, args.label)
    randomState = execution.initRandomState(args.randomState, directory)
    statsIterRandomStates = execution.initStatsIterRandomStates(statsIter,randomState)

    getDatabase = execution.getDatabaseFunction(args.name,args.type)


    DATASET, LABELS_DICTIONARY, datasetname = getDatabase(args.views, args.pathF, args.name, args.CL_nbClass,
                                             args.CL_classes, randomState, args.full, args.add_noise, args.noise_std)
    args.name = datasetname

    splits = execution.genSplits(DATASET.get("Labels").value, args.CL_split, statsIterRandomStates)

    multiclassLabels, labelsCombinations, indicesMulticlass = Multiclass.genMulticlassLabels(DATASET.get("Labels").value, multiclassMethod, splits)

    kFolds = execution.genKFolds(statsIter, args.CL_nbFolds, statsIterRandomStates)

    datasetFiles = Dataset.initMultipleDatasets(args.pathF, args.name, nbCores)

    # if not views:
    #     raise ValueError("Empty views list, modify selected views to match dataset " + args.views)

    views, viewsIndices, allViews = execution.initViews(DATASET, args.views)
    viewsDictionary = genViewsDictionnary(DATASET, views)
    nbViews = len(views)
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]

    metrics = [metric.split(":") for metric in args.CL_metrics]
    if metrics == [[""]]:
        metricsNames = [name for _, name, isPackage
                        in pkgutil.iter_modules(['./MonoMultiViewClassifiers/Metrics']) if not isPackage and name not in ["framework", "log_loss", "matthews_corrcoef", "roc_auc_score"]]
        metrics = [[metricName] for metricName in metricsNames]
        metrics = arangeMetrics(metrics, args.CL_metric_princ)
    for metricIndex, metric in enumerate(metrics):
        if len(metric) == 1:
            metrics[metricIndex] = [metric[0], None]

    benchmark = initBenchmark(CL_type, monoviewAlgos, multiviewAlgos, args)

    initKWARGS = initKWARGSFunc(args, benchmark)

    dataBaseTime = time.time() - start

    argumentDictionaries = initMonoviewExps(benchmark, viewsDictionary, NB_CLASS, initKWARGS)
    directories = execution.genDirecortiesNames(directory, statsIter)
    benchmarkArgumentDictionaries = execution.genArgumentDictionaries(LABELS_DICTIONARY, directories, multiclassLabels,
                                                                      labelsCombinations, indicesMulticlass,
                                                                      hyperParamSearch, args, kFolds,
                                                                      statsIterRandomStates, metrics,
                                                                      argumentDictionaries, benchmark, nbViews,
                                                                      views, viewsIndices)
    nbMulticlass = len(labelsCombinations)

    execBenchmark(nbCores, statsIter, nbMulticlass, benchmarkArgumentDictionaries, splits, directories,
                  directory, multiclassLabels, metrics, LABELS_DICTIONARY, NB_CLASS, DATASET)











    #
# def classifyOneIter_multicore(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices,
#                               kFolds,
#                               randomState, hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime, start,
#                               benchmark,
#                               views):
#     """Used to execute mono and multiview classification and result analysis for one random state
#      using multicore classification"""
#     resultsMonoview = []
#     labelsNames = LABELS_DICTIONARY.values()
#     np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")
#
#     resultsMonoview += [ExecMonoview_multicore(directory, args.name, labelsNames, classificationIndices, kFolds,
#                                                coreIndex, args.type, args.pathF, randomState,
#                                                hyperParamSearch=hyperParamSearch,
#                                                metrics=metrics, nIter=args.CL_HPS_iter,
#                                                **arguments)
#                         for arguments in argumentDictionaries["Monoview"]]
#     monoviewTime = time.time() - dataBaseTime - start
#
#     argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
#                                                   randomState, directory, resultsMonoview, classificationIndices)
#
#     resultsMultiview = []
#     resultsMultiview += [
#         ExecMultiview_multicore(directory, coreIndex, args.name, classificationIndices, kFolds, args.type,
#                                 args.pathF, LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
#                                 metrics=metrics, nIter=args.CL_HPS_iter, **arguments)
#         for arguments in argumentDictionaries["Multiview"]]
#     multiviewTime = time.time() - monoviewTime - dataBaseTime - start
#
#     labels = np.array(
#         [resultMonoview[1][3] for resultMonoview in resultsMonoview] + [resultMultiview[3] for resultMultiview in
#                                                                         resultsMultiview]).transpose()
#     DATASET = h5py.File(args.pathF + args.name + str(0) + ".hdf5", "r")
#     trueLabels = DATASET.get("Labels").value
#     times = [dataBaseTime, monoviewTime, multiviewTime]
#     results = (resultsMonoview, resultsMultiview)
#     labelAnalysis = analyzeLabels(labels, trueLabels, results, directory)
#     logging.debug("Start:\t Analyze Iteration Results")
#     resultAnalysis(benchmark, results, args.name, times, metrics, directory)
#     logging.debug("Done:\t Analyze Iteration Results")
#     globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
#     totalTime = time.time() - start
#     logging.info("Extraction time : " + str(int(dataBaseTime)) +
#                  "s, Monoview time : " + str(int(monoviewTime)) +
#                  "s, Multiview Time : " + str(int(multiviewTime)) +
#                  "s, Iteration Analysis Time : " + str(int(globalAnalysisTime)) +
#                  "s, Iteration Duration : " + str(int(totalTime)) + "s")
#     return results, labelAnalysis
#
#
# def classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices, kFolds,
#                     randomState, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
#                     benchmark, views):
#     """Used to execute mono and multiview classification and result analysis for one random state
#          classification"""
#     #TODO : Clarify this one
#
#
#     argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
#                                                   randomState, directory, resultsMonoview, classificationIndices)
#
#     resultsMultiview = []
#     if nbCores > 1:
#         nbExperiments = len(argumentDictionaries["Multiview"])
#         for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
#             resultsMultiview += Parallel(n_jobs=nbCores)(
#                 delayed(ExecMultiview_multicore)(directory, coreIndex, args.name, classificationIndices, kFolds,
#                                                  args.type,
#                                                  args.pathF,
#                                                  LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
#                                                  metrics=metrics, nIter=args.CL_HPS_iter,
#                                                  **argumentDictionaries["Multiview"][stepIndex * nbCores + coreIndex])
#                 for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores)))
#     else:
#         resultsMultiview = [
#             ExecMultiview(directory, DATASET, args.name, classificationIndices, kFolds, 1, args.type, args.pathF,
#                           LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
#                           metrics=metrics, nIter=args.CL_HPS_iter, **arguments) for arguments in
#             argumentDictionaries["Multiview"]]
#     multiviewTime = time.time() - monoviewTime - dataBaseTime - start
#     if nbCores > 1:
#         logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
#         datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
#         logging.debug("Start:\t Deleting datasets for multiprocessing")
#     labels = np.array(
#         [resultMonoview[1][3] for resultMonoview in resultsMonoview] + [resultMultiview[3] for resultMultiview in
#                                                                         resultsMultiview]).transpose()
#     trueLabels = DATASET.get("Labels").value
#     times = [dataBaseTime, monoviewTime, multiviewTime]
#     results = (resultsMonoview, resultsMultiview)
#     labelAnalysis = analyzeLabels(labels, trueLabels, results, directory)
#     logging.debug("Start:\t Analyze Iteration Results")
#     resultAnalysis(benchmark, results, args.name, times, metrics, directory)
#     logging.debug("Done:\t Analyze Iteration Results")
#     globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
#     totalTime = time.time() - start
#     logging.info("Extraction time : " + str(int(dataBaseTime)) +
#                  "s, Monoview time : " + str(int(monoviewTime)) +
#                  "s, Multiview Time : " + str(int(multiviewTime)) +
#                  "s, Iteration Analysis Time : " + str(int(globalAnalysisTime)) +
#                  "s, Iteration Duration : " + str(int(totalTime)) + "s")
#     return results, labelAnalysis
    #
    #
    #
    #
    #
    #
    #
    # if statsIter > 1:
    #     logging.debug("Start:\t Benchmark classification")
    #     for statIterIndex in range(statsIter):
    #         if not os.path.exists(os.path.dirname(directories[statIterIndex] + "train_labels.csv")):
    #             try:
    #                 os.makedirs(os.path.dirname(directories[statIterIndex] + "train_labels.csv"))
    #             except OSError as exc:
    #                 if exc.errno != errno.EEXIST:
    #                     raise
    #         trainIndices, testIndices = classificationIndices[statIterIndex]
    #         trainLabels = DATASET.get("Labels").value[trainIndices]
    #         np.savetxt(directories[statIterIndex] + "train_labels.csv", trainLabels, delimiter=",")
    #     if nbCores > 1:
    #         iterResults = []
    #         nbExperiments = statsIter*len(multiclassLabels)
    #         for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
    #             iterResults += (Parallel(n_jobs=nbCores)(
    #                 delayed(classifyOneIter_multicore)(LABELS_DICTIONARY, argumentDictionaries, 1,
    #                                                    directories[coreIndex + stepIndex * nbCores], args,
    #                                                    classificationIndices[coreIndex + stepIndex * nbCores],
    #                                                    kFolds[coreIndex + stepIndex * nbCores],
    #                                                    statsIterRandomStates[coreIndex + stepIndex * nbCores],
    #                                                    hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime,
    #                                                    start, benchmark,
    #                                                    views)
    #                 for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))
    #         logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
    #         datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
    #         logging.debug("Start:\t Deleting datasets for multiprocessing")
    #     else:
    #         iterResults = []
    #         for iterIndex in range(statsIter):
    #             if not os.path.exists(os.path.dirname(directories[iterIndex] + "train_labels.csv")):
    #                 try:
    #                     os.makedirs(os.path.dirname(directories[iterIndex] + "train_labels.csv"))
    #                 except OSError as exc:
    #                     if exc.errno != errno.EEXIST:
    #                         raise
    #             trainIndices, testIndices = classificationIndices[iterIndex]
    #             trainLabels = DATASET.get("Labels").value[trainIndices]
    #             np.savetxt(directories[iterIndex] + "train_labels.csv", trainLabels, delimiter=",")
    #             iterResults.append(
    #                 classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directories[iterIndex], args,
    #                                 classificationIndices[iterIndex], kFolds[iterIndex], statsIterRandomStates[iterIndex],
    #                                 hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start, benchmark,
    #                                 views))
    #     logging.debug("Done:\t Benchmark classification")
    #     logging.debug("Start:\t Global Results Analysis")
    #     classifiersIterResults = []
    #     iterLabelAnalysis = []
    #     for result in iterResults:
    #         classifiersIterResults.append(result[0])
    #         iterLabelAnalysis.append(result[1])
    #
    #     mono,multi = classifiersIterResults[0]
    #     classifiersNames = genNamesFromRes(mono, multi)
    #     analyzeIterLabels(iterLabelAnalysis, directory, classifiersNames)
    #     analyzeIterResults(classifiersIterResults, args.name, metrics, directory)
    #     logging.debug("Done:\t Global Results Analysis")
    #     totalDur = time.time() - start
    #     m, s = divmod(totalDur, 60)
    #     h, m = divmod(m, 60)
    #     d, h = divmod(h, 24)
    #     # print "%d_%02d_%02d" % (h, m, s)
    #     logging.info("Info:\t Total duration : " + str(d) + " days, " + str(h) + " hours, " + str(m) + " mins, " + str(
    #         int(s)) + "secs.")
    #
    # else:
    #     logging.debug("Start:\t Benchmark classification")
    #     if not os.path.exists(os.path.dirname(directories + "train_labels.csv")):
    #         try:
    #             os.makedirs(os.path.dirname(directories + "train_labels.csv"))
    #         except OSError as exc:
    #             if exc.errno != errno.EEXIST:
    #                 raise
    #     trainIndices, testIndices = classificationIndices
    #     trainLabels = DATASET.get("Labels").value[trainIndices]
    #     np.savetxt(directories + "train_labels.csv", trainLabels, delimiter=",")
    #     res, labelAnalysis = classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directories, args, classificationIndices,
    #                                          kFolds,
    #                                          statsIterRandomStates, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
    #                                          benchmark, views)
    #     logging.debug("Done:\t Benchmark classification")
    #     totalDur = time.time()-start
    #     m, s = divmod(totalDur, 60)
    #     h, m = divmod(m, 60)
    #     d, h = divmod(h, 24)
    #     # print "%d_%02d_%02d" % (h, m, s)
    #     logging.info("Info:\t Total duration : "+str(d)+ " days, "+str(h)+" hours, "+str(m)+" mins, "+str(int(s))+"secs.")
    #
    # if statsIter > 1:
    #     pass
