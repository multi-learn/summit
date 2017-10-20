# Import built-in modules
# import pdb;pdb.set_trace()
import pkgutil  # for TimeStamp in CSVFile
import os
import time
# import sys
import logging
import errno

# Import 3rd party modules
from joblib import Parallel, delayed
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure
import h5py

# Import own modules
from . import Multiview
# import Metrics
from . import MonoviewClassifiers
from .Multiview.ExecMultiview import ExecMultiview, ExecMultiview_multicore
from .Monoview.ExecClassifMonoView import ExecMonoview, ExecMonoview_multicore
from .Multiview import GetMultiviewDb as DB
from ResultAnalysis import resultAnalysis, analyzeLabels, analyzeIterResults, analyzeIterLabels, genNamesFromRes
from .utils import execution, Dataset

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def initBenchmark(args):
    """Used to create a list of all the algorithm packages names used for the benchmark
    Needs args.CL_type, args.CL_algos_multiview, args.MU_types, args.FU_types, args.FU_late_methods,
    args.FU_early_methods, args.CL_algos_monoview"""
    benchmark = {"Monoview": {}, "Multiview": {}}
    allMultiviewPackages = [name for _, name, isPackage
                            in pkgutil.iter_modules(['./MonoMultiViewClassifiers/Multiview/']) if isPackage]
    if args.CL_type == ["Benchmark"]:

        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['./MonoMultiViewClassifiers/MonoviewClassifiers'])
                            if (not isPackage)]
        benchmark["Monoview"] = allMonoviewAlgos
        benchmark["Multiview"] = dict((multiviewPackageName, "_") for multiviewPackageName in allMultiviewPackages)
        for multiviewPackageName in allMultiviewPackages:
            multiviewPackage = getattr(Multiview, multiviewPackageName)
            multiviewModule = getattr(multiviewPackage, multiviewPackageName+"Module")
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
                multiviewModule = getattr(multiviewPackage, multiviewPackageName+"Module")
                benchmark = multiviewModule.getBenchmark(benchmark, args=args)
    if "Monoview" in args.CL_type:
        if args.CL_algos_monoview == ['']:
            benchmark["Monoview"] = [name for _, name, isPackage in pkgutil.iter_modules(["./MonoMultiViewClassifiers/MonoviewClassifiers"])
                                     if not isPackage]

        else:
            benchmark["Monoview"] = args.CL_algos_monoview
    return benchmark


def initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, NB_CLASS, kwargsInit):
    if benchmark["Monoview"]:
        argumentDictionaries["Monoview"] = []
        for view in views:
            for classifier in benchmark["Monoview"]:
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


def initKWARGSFunc(args, benchmark):
    monoviewKWARGS = initMonoviewKWARGS(args, benchmark["Monoview"])
    return monoviewKWARGS


# def lateFusionSetArgs(views, viewsIndices, classes, method,
#                       classifiersNames, classifiersConfig, fusionMethodConfig):
#     arguments = {"CL_type": "Fusion",
#                  "views": views,
#                  "NB_VIEW": len(views),
#                  "viewsIndices": viewsIndices,
#                  "NB_CLASS": len(classes),
#                  "LABELS_NAMES": args.CL_classes,
#                  "FusionKWARGS": {"fusionType": "LateFusion", "fusionMethod": method,
#                                   "classifiersNames": classifiersNames,
#                                   "classifiersConfigs": classifiersConfig,
#                                   'fusionMethodConfig': fusionMethodConfig,
#                                   "nbView": (len(viewsIndices))}}
#     return arguments


def initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries, randomState, directory,
                           resultsMonoview, classificationIndices):
    multiviewArguments = []
    if "Multiview" in benchmark:
        for multiviewAlgoName in benchmark["Multiview"]:
            multiviewPackage = getattr(Multiview, multiviewAlgoName)
            mutliviewModule = getattr(multiviewPackage, multiviewAlgoName+"Module")
            multiviewArguments += mutliviewModule.getArgs(args, benchmark, views, viewsIndices, randomState, directory,
                                                          resultsMonoview, classificationIndices)
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


def classifyOneIter_multicore(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices,
                              kFolds,
                              randomState, hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime, start,
                              benchmark,
                              views):
    resultsMonoview = []
    np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")
    labelsNames = LABELS_DICTIONARY.values()
    resultsMonoview += [ExecMonoview_multicore(directory, args.name, labelsNames, classificationIndices, kFolds,
                                               coreIndex, args.type, args.pathF, randomState,
                                               hyperParamSearch=hyperParamSearch,
                                               metrics=metrics, nIter=args.CL_GS_iter,
                                               **arguments)
                        for arguments in argumentDictionaries["Monoview"]]
    monoviewTime = time.time() - dataBaseTime - start

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)

    resultsMultiview = []
    resultsMultiview += [
        ExecMultiview_multicore(directory, coreIndex, args.name, classificationIndices, kFolds, args.type,
                                args.pathF, LABELS_DICTIONARY, randomState, hyperParamSearch=hyperParamSearch,
                                metrics=metrics, nIter=args.CL_GS_iter, **arguments)
        for arguments in argumentDictionaries["Multiview"]]
    multiviewTime = time.time() - monoviewTime - dataBaseTime - start

    labels = np.array(
        [resultMonoview[1][3] for resultMonoview in resultsMonoview] + [resultMultiview[3] for resultMultiview in
                                                                        resultsMultiview]).transpose()
    DATASET = h5py.File(args.pathF + args.name + str(0) + ".hdf5", "r")
    trueLabels = DATASET.get("Labels").value
    times = [dataBaseTime, monoviewTime, multiviewTime]
    results = (resultsMonoview, resultsMultiview)
    labelAnalysis = analyzeLabels(labels, trueLabels, results, directory)
    logging.debug("Start:\t Analyze Global Results for iteration")
    resultAnalysis(benchmark, results, args.name, times, metrics, directory)
    logging.debug("Done:\t Analyze Global Results for iteration")
    globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
    totalTime = time.time() - start
    logging.info("Extraction time : " + str(dataBaseTime) +
                 "s, Monoview time : " + str(monoviewTime) +
                 "s, Multiview Time : " + str(multiviewTime) +
                 "s, Global Analysis Time : " + str(globalAnalysisTime) +
                 "s, Total Duration : " + str(totalTime) + "s")
    return results, labelAnalysis


def classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices, kFolds,
                    randomState, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
                    benchmark, views):

    np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")
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

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)

    resultsMultiview = []
    if nbCores > 1:
        nbExperiments = len(argumentDictionaries["Multiview"])
        for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
            resultsMultiview += Parallel(n_jobs=nbCores)(
                delayed(ExecMultiview_multicore)(directory, coreIndex, args.name, classificationIndices, kFolds,
                                                 args.type,
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
    labelAnalysis = analyzeLabels(labels, trueLabels, results, directory)
    logging.debug("Start:\t Analyze Global Results")
    resultAnalysis(benchmark, results, args.name, times, metrics, directory)
    logging.debug("Done:\t Analyze Global Results")
    globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
    totalTime = time.time() - start
    logging.info("Extraction time : " + str(dataBaseTime) +
                 "s, Monoview time : " + str(monoviewTime) +
                 "s, Multiview Time : " + str(multiviewTime) +
                 "s, Global Analysis Time : " + str(globalAnalysisTime) +
                 "s, Total Duration : " + str(totalTime) + "s")
    return results, labelAnalysis


# _______________ #
# __ EXECUTION __ #
# _______________ #
def execClassif(arguments):
    # import pdb;pdb.set_trace()
    start = time.time()
    args = execution.parseTheArgs(arguments)

    os.nice(args.nice)
    nbCores = args.CL_cores
    statsIter = args.CL_statsiter
    hyperParamSearch = args.CL_HPS_type

    directory = execution.initLogFile(args)
    randomState = execution.initRandomState(args.randomState, directory)
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
    classificationIndices = execution.genSplits(statsIter, datasetLength, DATASET, args.CL_split, statsIterRandomStates)

    kFolds = execution.genKFolds(statsIter, args.CL_nbFolds, statsIterRandomStates)

    datasetFiles = Dataset.initMultipleDatasets(args, nbCores)

    views, viewsIndices, allViews = execution.initViews(DATASET, args)
    if not views:
        raise ValueError, "Empty views list, modify selected views to match dataset " + args.views

    NB_VIEW = len(views)
    NB_CLASS = DATASET.get("Metadata").attrs["nbClass"]

    metrics = [metric.split(":") for metric in args.CL_metrics]
    if metrics == [[""]]:
        metricsNames = [name for _, name, isPackage
                        in pkgutil.iter_modules(['./MonoMultiViewClassifiers/Metrics']) if not isPackage and name != "log_loss"]
        metrics = [[metricName] for metricName in metricsNames]
        metrics = arangeMetrics(metrics, args.CL_metric_princ)
    for metricIndex, metric in enumerate(metrics):
        if len(metric) == 1:
            metrics[metricIndex] = [metric[0], None]

    logging.info("Start:\t Finding all available mono- & multiview algorithms")

    benchmark = initBenchmark(args)

    initKWARGS = initKWARGSFunc(args, benchmark)

    dataBaseTime = time.time() - start

    argumentDictionaries = {"Monoview": [], "Multiview": []}
    argumentDictionaries = initMonoviewArguments(benchmark, argumentDictionaries, views, allViews, NB_CLASS,
                                                 initKWARGS)
    directories = execution.genDirecortiesNames(directory, statsIter)

    if statsIter > 1:
        for statIterIndex in range(statsIter):
            if not os.path.exists(os.path.dirname(directories[statIterIndex] + "train_labels.csv")):
                try:
                    os.makedirs(os.path.dirname(directories[statIterIndex] + "train_labels.csv"))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            trainIndices, testIndices = classificationIndices[statIterIndex]
            trainLabels = DATASET.get("Labels").value[trainIndices]
            np.savetxt(directories[statIterIndex] + "train_labels.csv", trainLabels, delimiter=",")
        if nbCores > 1:
            iterResults = []
            nbExperiments = statsIter
            for stepIndex in range(int(math.ceil(float(nbExperiments) / nbCores))):
                iterResults += (Parallel(n_jobs=nbCores)(
                    delayed(classifyOneIter_multicore)(LABELS_DICTIONARY, argumentDictionaries, 1,
                                                       directories[coreIndex + stepIndex * nbCores], args,
                                                       classificationIndices[coreIndex + stepIndex * nbCores],
                                                       kFolds[coreIndex + stepIndex * nbCores],
                                                       statsIterRandomStates[coreIndex + stepIndex * nbCores],
                                                       hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime,
                                                       start, benchmark,
                                                       views)
                    for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))
            logging.debug("Start:\t Deleting " + str(nbCores) + " temporary datasets for multiprocessing")
            datasetFiles = DB.deleteHDF5(args.pathF, args.name, nbCores)
            logging.debug("Start:\t Deleting datasets for multiprocessing")
        else:
            iterResults = []
            for iterIndex in range(statsIter):
                if not os.path.exists(os.path.dirname(directories[iterIndex] + "train_labels.csv")):
                    try:
                        os.makedirs(os.path.dirname(directories[iterIndex] + "train_labels.csv"))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                trainIndices, testIndices = classificationIndices[iterIndex]
                trainLabels = DATASET.get("Labels").value[trainIndices]
                np.savetxt(directories[iterIndex] + "train_labels.csv", trainLabels, delimiter=",")
                iterResults.append(
                    classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directories[iterIndex], args,
                                    classificationIndices[iterIndex], kFolds[iterIndex], statsIterRandomStates[iterIndex],
                                    hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start, benchmark,
                                    views))
        classifiersIterResults = []
        iterLabelAnalysis = []
        for result in iterResults:
            classifiersIterResults.append(result[0])
            iterLabelAnalysis.append(result[1])

        mono,multi = classifiersIterResults[0]
        classifiersNames = genNamesFromRes(mono, multi)
        analyzeIterLabels(iterLabelAnalysis, directory, classifiersNames)
        analyzeIterResults(classifiersIterResults, args.name, metrics, directory)

    else:
        if not os.path.exists(os.path.dirname(directories + "train_labels.csv")):
            try:
                os.makedirs(os.path.dirname(directories + "train_labels.csv"))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        trainIndices, testIndices = classificationIndices
        trainLabels = DATASET.get("Labels").value[trainIndices]
        np.savetxt(directories + "train_labels.csv", trainLabels, delimiter=",")
        res, labelAnalysis = classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directories, args, classificationIndices,
                              kFolds,
                              statsIterRandomStates, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
                              benchmark, views)

    if statsIter > 1:
        pass
