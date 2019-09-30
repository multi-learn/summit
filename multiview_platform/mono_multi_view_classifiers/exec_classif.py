import errno
import logging
import math
import os
import pkgutil
import time

import matplotlib
import itertools
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

matplotlib.use(
    'Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure

# Import own modules
from . import monoview_classifiers
from . import multiview_classifiers
from .multiview.exec_multiview import ExecMultiview, ExecMultiview_multicore
from .monoview.exec_classif_mono_view import ExecMonoview, ExecMonoview_multicore
from .utils import get_multiview_db as DB
from .result_analysis import \
    getResults, plot_results_noise  # resultAnalysis, analyzeLabels, analyzeIterResults, analyzeIterLabels, genNamesFromRes,
from .utils import execution, dataset, multiclass, configuration

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def initBenchmark(CL_type, monoviewAlgos, multiviewAlgos, args):
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
    benchmark = {"monoview": {}, "multiview": {}}
    allMultiviewPackages = [name for _, name, isPackage
                            in pkgutil.iter_modules(
            ['./mono_multi_view_classifiers/multiview_classifiers/']) if isPackage]

    if "monoview" in CL_type:
        if monoviewAlgos == ['all']:
            benchmark["monoview"] = [name for _, name, isPackage in
                                     pkgutil.iter_modules([
                                                              "./mono_multi_view_classifiers/monoview_classifiers"])
                                     if not isPackage]

        else:
            benchmark["monoview"] = monoviewAlgos

    if "multiview" in CL_type:
        benchmark["multiview"] = [name for _, name, isPackage in
                                 pkgutil.iter_modules([
                                     "./mono_multi_view_classifiers/multiview_classifiers"])
                                 if not isPackage]
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
        if datasetName[:4] == "View":
            viewName = DATASET.get(datasetName).attrs["name"]
            if type(viewName) == bytes:
                viewName = viewName.decode("utf-8")
            if viewName in views:
                viewsDictionary[viewName] = int(datasetName[4:])

    return viewsDictionary


def init_argument_dictionaries(benchmark, views_dictionary,
                                                    nb_class, init_kwargs):
    argument_dictionaries = {"monoview": [], "multiview": []}
    if benchmark["monoview"]:
        argument_dictionaries["monoview"] = init_monoview_exps(
                                                   benchmark["monoview"],
                                                   views_dictionary,
                                                   nb_class,
                                                   init_kwargs["monoview"])
    if benchmark["multiview"]:
        argument_dictionaries["multiview"] = init_multiview_exps(benchmark["multiview"],
                                                   views_dictionary,
                                                   nb_class,
                                                   init_kwargs["multiview"])
    return argument_dictionaries


def init_multiview_exps(classifier_names, views_dictionary, nb_class, kwargs_init):
    multiview_arguments = []
    for classifier_name in classifier_names:
        if multiple_args(classifier_name, kwargs_init):
            multiview_arguments += gen_multiple_args_dictionnaries(nb_class,
                                                                  kwargs_init,
                                                                  classifier_name,
                                                                  views_dictionary=views_dictionary,
                                                                  framework="multiview")
        else:
            multiview_arguments += [gen_single_multiview_arg_dictionary(classifier_name,
                                                                        kwargs_init,
                                                                        nb_class,
                                                                        views_dictionary=views_dictionary)]
    return multiview_arguments


def init_monoview_exps(classifier_names,
                       views_dictionary, nb_class, kwargs_init):
    r"""Used to add each monoview exeperience args to the list of monoview experiences args.

    First this function will check if the benchmark need mono- or/and multiview algorithms and adds to the right
    dictionary the asked algorithms. If none is asked by the user, all will be added.

    If the keyword `"Benchmark"` is used, all mono- and multiview algorithms will be added.

    Parameters
    ----------
    classifier_names : dictionary
        All types of monoview and multiview experiments that have to be benchmarked
    argument_dictionaries : dictionary
        Maps monoview and multiview experiments arguments.
    viewDictionary : dictionary
        Maps the view names to their index in the HDF5 dataset
    nb_class : integer
        Number of different labels in the classification

    Returns
    -------
    benchmark : Dictionary of dictionaries
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    """
    monoview_arguments = []
    for view_name, view_index in views_dictionary.items():
        for classifier in classifier_names:
            if multiple_args(classifier, kwargs_init):
                monoview_arguments += gen_multiple_args_dictionnaries(nb_class,
                                                                      kwargs_init,
                                                                      classifier,
                                                                      view_name,
                                                                      view_index)
            else:
                arguments = gen_single_monoview_arg_dictionary(classifier,
                                                               kwargs_init,
                                                               nb_class,
                                                               view_index,
                                                               view_name)
                monoview_arguments.append(arguments)
    return monoview_arguments


def gen_single_monoview_arg_dictionary(classifier_name, arguments, nb_class,
                                       view_index, view_name):
    return {classifier_name: dict((key, value[0]) for key, value in arguments[
                                                 classifier_name].items()),
            "view_name": view_name,
            "view_index": view_index,
            "classifier_name": classifier_name,
            "nb_class": nb_class}


def gen_single_multiview_arg_dictionary(classifier_name,arguments,nb_class,
                                        views_dictionary=None):
    return {"classifier_name": classifier_name,
            "view_names": list(views_dictionary.keys()),
            'view_indices': list(views_dictionary.values()),
            "nb_class": nb_class,
            "labels_names": None,
            classifier_name: dict((key, value[0]) for key, value in arguments[
                                                 classifier_name].items())
            }


def multiple_args(classifier, kwargsInit):
    listed_args = [type(value) == list and len(value)>1 for key, value in
                   kwargsInit[classifier].items()]
    if True in listed_args:
        return True
    else: 
        return False


def gen_multiple_kwargs_combinations(clKWARGS):
    values = list(clKWARGS.values())
    listed_values = [[_] if type(_) is not list else _ for _ in values]
    values_cartesian_prod = [_ for _ in itertools.product(*listed_values)]
    keys = clKWARGS.keys()
    kwargs_combination = [dict((key, value) for key, value in zip(keys, values))
                          for values in values_cartesian_prod]

    reduce_dict = {DecisionTreeClassifier: "DT", }
    reduced_listed_values = [
        [_ if type(_) not in reduce_dict else reduce_dict[type(_)] for _ in
         list_] for list_ in listed_values]
    reduced_values_cartesian_prod = [_ for _ in itertools.product(*reduced_listed_values)]
    reduced_kwargs_combination = [dict((key, value) for key, value in zip(keys, values))
                          for values in reduced_values_cartesian_prod]
    return kwargs_combination, reduced_kwargs_combination


def gen_multiple_args_dictionnaries(nb_class, kwargsInit, classifier,
                                    view_name=None, view_index=None, views_indices=None,
                                    framework="monoview"):
    multiple_kwargs_list, reduced_multiple_kwargs_list = gen_multiple_kwargs_combinations(kwargsInit[classifier])
    multiple_kwargs_dict = dict(
        (classifier+"_"+"_".join(map(str,list(reduced_dictionary.values()))), dictionary)
        for reduced_dictionary, dictionary in zip(reduced_multiple_kwargs_list, multiple_kwargs_list ))
    args_dictionnaries = [gen_single_monoview_arg_dictionary(classifier_name,
                                                              arguments,
                                                              nb_class,
                                                              view_index=view_index,
                                                              view_name=view_name)
                           if framework=="monoview" else
                           gen_single_multiview_arg_dictionary(classifier_name,
                                                            arguments,
                                                            nb_class,
                                                            views_indices=views_indices)
                           for classifier_name, arguments
                           in multiple_kwargs_dict.items()]
    return args_dictionnaries


def init_monoview_kwargs(args, classifiersNames):
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

    logging.debug("Start:\t Initializing monoview classifiers arguments")
    monoviewKWARGS = {}
    for classifiersName in classifiersNames:
        try:
            getattr(monoview_classifiers, classifiersName)
        except AttributeError:
            raise AttributeError(
                classifiersName + " is not implemented in monoview_classifiers, "
                                  "please specify the name of the file in monoview_classifiers")
        monoviewKWARGS[
            classifiersName] = args[classifiersName]
    logging.debug("Done:\t Initializing monoview classifiers arguments")

    return monoviewKWARGS


def initKWARGSFunc(args, benchmark):
    monoview_kwargs = init_monoview_kwargs(args, benchmark["monoview"])
    multiview_kwargs = init_multiview_kwargs(args, benchmark["multiview"])
    kwargs = {"monoview":monoview_kwargs, "multiview":multiview_kwargs}
    return kwargs


def init_multiview_kwargs(args, classifiers_names):
    logging.debug("Start:\t Initializing multiview classifiers arguments")
    multiview_kwargs = {}
    for classifiers_name in classifiers_names:
        try:
            getattr(multiview_classifiers, classifiers_name)
        except AttributeError:
            raise AttributeError(
                classifiers_name + " is not implemented in mutliview_classifiers, "
                                  "please specify the name of the coressponding .py "
                                   "file in mutliview_classifiers")
        multiview_kwargs[classifiers_name] = args[classifiers_name]
    logging.debug("Done:\t Initializing multiview classifiers arguments")
    return multiview_kwargs


def initMultiviewArguments(args, benchmark, views, viewsIndices,
                           argumentDictionaries, randomState, directory,
                           resultsMonoview, classificationIndices):
    """Used to add each monoview exeperience args to the list of monoview experiences args"""
    logging.debug("Start:\t Initializing multiview classifiers arguments")
    multiviewArguments = []
    if "multiview" in benchmark:
        for multiviewAlgoName in benchmark["multiview"]:
            mutliviewModule = getattr(multiview_classifiers,
                                      multiviewAlgoName)

            multiviewArguments += mutliviewModule.getArgs(args, benchmark,
                                                          views, viewsIndices,
                                                          randomState,
                                                          directory,
                                                          resultsMonoview,
                                                          classificationIndices)
    argumentDictionaries["multiview"] = multiviewArguments
    logging.debug("Start:\t Initializing multiview classifiers arguments")
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


def benchmarkInit(directory, classificationIndices, labels, LABELS_DICTIONARY,
                  kFolds):
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
    np.savetxt(directory + "train_indices.csv", classificationIndices[0],
               delimiter=",")
    resultsMonoview = []
    folds = kFolds.split(np.arange(len(trainLabels)), trainLabels)
    minFoldLen = int(len(trainLabels) / kFolds.n_splits)
    for foldIndex, (trainCVIndices, testCVIndices) in enumerate(folds):
        fileName = directory + "/folds/test_labels_fold_" + str(
            foldIndex) + ".csv"
        if not os.path.exists(os.path.dirname(fileName)):
            try:
                os.makedirs(os.path.dirname(fileName))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.savetxt(fileName, trainLabels[testCVIndices[:minFoldLen]],
                   delimiter=",")
    labelsNames = list(LABELS_DICTIONARY.values())
    logging.debug("Done:\t Benchmark initialization")
    return resultsMonoview, labelsNames


def execOneBenchmark(coreIndex=-1, LABELS_DICTIONARY=None, directory=None,
                     classificationIndices=None, args=None,
                     kFolds=None, randomState=None, hyperParamSearch=None,
                     metrics=None, argumentDictionaries=None,
                     benchmark=None, views=None, viewsIndices=None, flag=None,
                     labels=None,
                     ExecMonoview_multicore=ExecMonoview_multicore,
                     ExecMultiview_multicore=ExecMultiview_multicore,
                     initMultiviewArguments=initMultiviewArguments):
    """Used to run a benchmark using one core. ExecMonoview_multicore, initMultiviewArguments and
     ExecMultiview_multicore args are only used for tests"""

    resultsMonoview, labelsNames = benchmarkInit(directory,
                                                 classificationIndices, labels,
                                                 LABELS_DICTIONARY, kFolds)

    logging.debug("Start:\t monoview benchmark")
    resultsMonoview += [
        ExecMonoview_multicore(directory, args["Base"]["name"], labelsNames,
                               classificationIndices, kFolds,
                               coreIndex, args["Base"]["type"], args["Base"]["pathf"], randomState,
                               labels,
                               hyperParamSearch=hyperParamSearch,
                               metrics=metrics,
                               nIter=args["Classification"]["hps_iter"], **argument)
        for argument in argumentDictionaries["Monoview"]]
    logging.debug("Done:\t monoview benchmark")

    logging.debug("Start:\t multiview arguments initialization")
    # argumentDictionaries = initMultiviewArguments(args, benchmark, views,
    #                                               viewsIndices,
    #                                               argumentDictionaries,
    #                                               randomState, directory,
    #                                               resultsMonoview,
    #                                               classificationIndices)
    logging.debug("Done:\t multiview arguments initialization")

    logging.debug("Start:\t multiview benchmark")
    resultsMultiview = [
        ExecMultiview_multicore(directory, coreIndex, args["Base"]["name"],
                                classificationIndices, kFolds, args["Base"]["type"],
                                args["Base"]["pathf"], LABELS_DICTIONARY, randomState,
                                labels, hyperParamSearch=hyperParamSearch,
                                metrics=metrics, nIter=args["Classification"]["hps_iter"],
                                **arguments)
        for arguments in argumentDictionaries["multiview"]]
    logging.debug("Done:\t multiview benchmark")

    return [flag, resultsMonoview + resultsMultiview]


def execOneBenchmark_multicore(nbCores=-1, LABELS_DICTIONARY=None,
                               directory=None, classificationIndices=None,
                               args=None,
                               kFolds=None, randomState=None,
                               hyperParamSearch=None, metrics=None,
                               argumentDictionaries=None,
                               benchmark=None, views=None, viewsIndices=None,
                               flag=None, labels=None,
                               ExecMonoview_multicore=ExecMonoview_multicore,
                               ExecMultiview_multicore=ExecMultiview_multicore,
                               initMultiviewArguments=initMultiviewArguments):
    """Used to run a benchmark using multiple cores. ExecMonoview_multicore, initMultiviewArguments and
     ExecMultiview_multicore args are only used for tests"""

    resultsMonoview, labelsNames = benchmarkInit(directory,
                                                 classificationIndices, labels,
                                                 LABELS_DICTIONARY, kFolds)

    logging.debug("Start:\t monoview benchmark")
    nbExperiments = len(argumentDictionaries["monoview"])
    nbMulticoreToDo = int(math.ceil(float(nbExperiments) / nbCores))
    for stepIndex in range(nbMulticoreToDo):
        resultsMonoview += (Parallel(n_jobs=nbCores)(
            delayed(ExecMonoview_multicore)(directory, args["Base"]["name"], labelsNames,
                                            classificationIndices, kFolds,
                                            coreIndex, args["Base"]["type"], args["Base"]["pathf"],
                                            randomState, labels,
                                            hyperParamSearch=hyperParamSearch,
                                            metrics=metrics,
                                            nIter=args["Classification"]["hps_iter"],
                                            **argumentDictionaries["monoview"][
                                                coreIndex + stepIndex * nbCores])
            for coreIndex in
            range(min(nbCores, nbExperiments - stepIndex * nbCores))))
    logging.debug("Done:\t monoview benchmark")

    logging.debug("Start:\t multiview arguments initialization")
    # argumentDictionaries = initMultiviewArguments(args, benchmark, views,
    #                                               viewsIndices,
    #                                               argumentDictionaries,
    #                                               randomState, directory,
    #                                               resultsMonoview,
    #                                               classificationIndices)
    logging.debug("Done:\t multiview arguments initialization")

    logging.debug("Start:\t multiview benchmark")
    resultsMultiview = []
    nbExperiments = len(argumentDictionaries["multiview"])
    nbMulticoreToDo = int(math.ceil(float(nbExperiments) / nbCores))
    for stepIndex in range(nbMulticoreToDo):
        resultsMultiview += Parallel(n_jobs=nbCores)(
            delayed(ExecMultiview_multicore)(directory, coreIndex, args["Base"]["name"],
                                             classificationIndices, kFolds,
                                             args["Base"]["type"], args["Base"]["pathf"],
                                             LABELS_DICTIONARY, randomState,
                                             labels,
                                             hyperParamSearch=hyperParamSearch,
                                             metrics=metrics,
                                             nIter=args["Classification"]["hps_iter"],
                                             **
                                             argumentDictionaries["multiview"][
                                                 stepIndex * nbCores + coreIndex])
            for coreIndex in
            range(min(nbCores, nbExperiments - stepIndex * nbCores)))
    logging.debug("Done:\t multiview benchmark")

    return [flag, resultsMonoview + resultsMultiview]


def execOneBenchmarkMonoCore(DATASET=None, LABELS_DICTIONARY=None,
                             directory=None, classificationIndices=None,
                             args=None,
                             kFolds=None, randomState=None,
                             hyperParamSearch=None, metrics=None,
                             argumentDictionaries=None,
                             benchmark=None, views=None, viewsIndices=None,
                             flag=None, labels=None,
                             ExecMonoview_multicore=ExecMonoview_multicore,
                             ExecMultiview_multicore=ExecMultiview_multicore,
                             initMultiviewArguments=initMultiviewArguments):
    resultsMonoview, labelsNames = benchmarkInit(directory,
                                                 classificationIndices, labels,
                                                 LABELS_DICTIONARY, kFolds)
    logging.debug("Start:\t monoview benchmark")
    for arguments in argumentDictionaries["monoview"]:
        X = DATASET.get("View" + str(arguments["view_index"]))
        Y = labels
        resultsMonoview += [
            ExecMonoview(directory, X, Y, args["Base"]["name"], labelsNames,
                         classificationIndices, kFolds,
                         1, args["Base"]["type"], args["Base"]["pathf"], randomState,
                         hyperParamSearch=hyperParamSearch, metrics=metrics,
                         nIter=args["Classification"]["hps_iter"], **arguments)]
    logging.debug("Done:\t monoview benchmark")

    logging.debug("Start:\t multiview arguments initialization")

    # argumentDictionaries = initMultiviewArguments(args, benchmark, views,
    #                                               viewsIndices,
    #                                               argumentDictionaries,
    #                                               randomState, directory,
    #                                               resultsMonoview,
    #                                               classificationIndices)
    logging.debug("Done:\t multiview arguments initialization")

    logging.debug("Start:\t multiview benchmark")
    resultsMultiview = []
    for arguments in argumentDictionaries["multiview"]:
        resultsMultiview += [
            ExecMultiview(directory, DATASET, args["Base"]["name"], classificationIndices,
                          kFolds, 1, args["Base"]["type"],
                          args["Base"]["pathf"], LABELS_DICTIONARY, randomState, labels,
                          hyperParamSearch=hyperParamSearch,
                          metrics=metrics, nIter=args["Classification"]["hps_iter"], **arguments)]
    logging.debug("Done:\t multiview benchmark")

    return [flag, resultsMonoview + resultsMultiview]


def execBenchmark(nbCores, statsIter, nbMulticlass,
                  benchmarkArgumentsDictionaries, classificationIndices,
                  directories,
                  directory, multiClassLabels, metrics, labelsDictionary,
                  nbLabels, DATASET,
                  execOneBenchmark=execOneBenchmark,
                  execOneBenchmark_multicore=execOneBenchmark_multicore,
                  execOneBenchmarkMonoCore=execOneBenchmarkMonoCore,
                  getResults=getResults, delete=DB.deleteHDF5):
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
        metrics that will be used to evaluate the algorithms performance.
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
                                                      **
                                                      benchmarkArgumentsDictionaries[
                                                          coreIndex + stepIndex * nbCores])
                                                     for coreIndex in range(
                    min(nbCores, nbExpsToDo - stepIndex * nbCores))))
        else:
            results += [execOneBenchmark_multicore(nbCores=nbCores, **
            benchmarkArgumentsDictionaries[0])]
    else:
        for arguments in benchmarkArgumentsDictionaries:
            results += [execOneBenchmarkMonoCore(DATASET=DATASET, **arguments)]
    logging.debug("Done:\t Executing all the needed biclass benchmarks")

    # Do everything with flagging
    nbExamples = len(classificationIndices[0][0]) + len(
        classificationIndices[0][1])
    multiclassGroundTruth = DATASET.get("Labels").value
    logging.debug("Start:\t Analyzing predictions")
    results_mean_stds = getResults(results, statsIter, nbMulticlass,
                                   benchmarkArgumentsDictionaries,
                                   multiclassGroundTruth,
                                   metrics,
                                   classificationIndices,
                                   directories,
                                   directory,
                                   labelsDictionary,
                                   nbExamples,
                                   nbLabels)
    logging.debug("Done:\t Analyzing predictions")
    delete(benchmarkArgumentsDictionaries, nbCores, DATASET)
    return results_mean_stds


def execClassif(arguments):
    """Main function to execute the benchmark"""
    start = time.time()
    args = execution.parseTheArgs(arguments)
    args = configuration.get_the_args(args.path_config)
    os.nice(args["Base"]["nice"])
    nbCores = args["Base"]["nb_cores"]
    if nbCores == 1:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    statsIter = args["Classification"]["stats_iter"]
    hyperParamSearch = args["Classification"]["hps_type"]
    multiclassMethod = args["Classification"]["multiclass_method"]
    CL_type = args["Classification"]["type"]
    monoviewAlgos = args["Classification"]["algos_monoview"]
    multiviewAlgos = args["Classification"]["algos_multiview"]
    dataset_list = execution.find_dataset_names(args["Base"]["pathf"],
                                                args["Base"]["type"],
                                                args["Base"]["name"])
    if not args["Base"]["add_noise"]:
        args["Base"]["noise_std"]=[0.0]

    for dataset_name in dataset_list:
        noise_results = []
        for noise_std in args["Base"]["noise_std"]:

            directory = execution.initLogFile(dataset_name, args["Base"]["views"], args["Classification"]["type"],
                                              args["Base"]["log"], args["Base"]["debug"], args["Base"]["label"],
                                              args["Base"]["res_dir"], args["Base"]["add_noise"], noise_std)
            randomState = execution.initRandomState(args["Base"]["random_state"], directory)
            statsIterRandomStates = execution.initStatsIterRandomStates(statsIter,
                                                                        randomState)

            getDatabase = execution.getDatabaseFunction(dataset_name, args["Base"]["type"])

            DATASET, LABELS_DICTIONARY, datasetname = getDatabase(args["Base"]["views"],
                                                                  args["Base"]["pathf"], dataset_name,
                                                                  args["Classification"]["nb_class"],
                                                                  args["Classification"]["classes"],
                                                                  randomState,
                                                                  args["Base"]["full"],
                                                                  args["Base"]["add_noise"],
                                                                  noise_std)
            args["Base"]["name"] = datasetname

            splits = execution.genSplits(DATASET.get("Labels").value, args["Classification"]["split"],
                                         statsIterRandomStates)

            multiclassLabels, labelsCombinations, indicesMulticlass = multiclass.genMulticlassLabels(
                DATASET.get("Labels").value, multiclassMethod, splits)

            kFolds = execution.genKFolds(statsIter, args["Classification"]["nb_folds"],
                                         statsIterRandomStates)

            datasetFiles = dataset.init_multiple_datasets(args["Base"]["pathf"], args["Base"]["name"], nbCores)


            views, viewsIndices, allViews = execution.initViews(DATASET, args["Base"]["views"])
            viewsDictionary = genViewsDictionnary(DATASET, views)
            nbViews = len(views)
            NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]

            metrics = [metric.split(":") for metric in args["Classification"]["metrics"]]
            if metrics == [["all"]]:
                metricsNames = [name for _, name, isPackage
                                in pkgutil.iter_modules(
                        ['./mono_multi_view_classifiers/metrics']) if
                                not isPackage and name not in ["framework", "log_loss",
                                                               "matthews_corrcoef",
                                                               "roc_auc_score"]]
                metrics = [[metricName] for metricName in metricsNames]
                metrics = arangeMetrics(metrics, args["Classification"]["metric_princ"])
            for metricIndex, metric in enumerate(metrics):
                if len(metric) == 1:
                    metrics[metricIndex] = [metric[0], None]

            benchmark = initBenchmark(CL_type, monoviewAlgos, multiviewAlgos, args)
            initKWARGS = initKWARGSFunc(args, benchmark)
            dataBaseTime = time.time() - start
            argumentDictionaries = init_argument_dictionaries(benchmark, viewsDictionary,
                                                    NB_CLASS, initKWARGS)
            # argumentDictionaries = initMonoviewExps(benchmark, viewsDictionary,
            #                                         NB_CLASS, initKWARGS)
            directories = execution.genDirecortiesNames(directory, statsIter)
            benchmarkArgumentDictionaries = execution.genArgumentDictionaries(
                LABELS_DICTIONARY, directories, multiclassLabels,
                labelsCombinations, indicesMulticlass,
                hyperParamSearch, args, kFolds,
                statsIterRandomStates, metrics,
                argumentDictionaries, benchmark, nbViews,
                views, viewsIndices)
            nbMulticlass = len(labelsCombinations)
            results_mean_stds = execBenchmark(nbCores, statsIter, nbMulticlass,
                                                  benchmarkArgumentDictionaries, splits, directories,
                                                  directory, multiclassLabels, metrics, LABELS_DICTIONARY,
                                                  NB_CLASS, DATASET)
            noise_results.append([noise_std, results_mean_stds])
            plot_results_noise(directory, noise_results, metrics[0][0], dataset_name)


