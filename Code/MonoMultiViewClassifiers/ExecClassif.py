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
from .ResultAnalysis import resultAnalysis, analyzeLabels, analyzeIterResults, analyzeIterLabels, genNamesFromRes
from .utils import execution, Dataset, Multiclass
from . import Metrics

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def initBenchmark(args):
    """Used to create a list of all the algorithm packages names used for the benchmark
    Needs args.CL_type, args.CL_algos_multiview, args.MU_types, args.FU_types, args.FU_late_methods,
    args.FU_early_methods, args.CL_algos_monoview"""
    benchmark = {"Monoview": {}, "Multiview": {}}
    allMultiviewPackages = [name for _, name, isPackage
                            in pkgutil.iter_modules(['./MonoMultiViewClassifiers/MultiviewClassifiers/']) if isPackage]
    if args.CL_type == ["Benchmark"]:

        allMonoviewAlgos = [name for _, name, isPackage in
                            pkgutil.iter_modules(['./MonoMultiViewClassifiers/MonoviewClassifiers'])
                            if (not isPackage)]
        benchmark["Monoview"] = allMonoviewAlgos
        benchmark["Multiview"] = dict((multiviewPackageName, "_") for multiviewPackageName in allMultiviewPackages)
        for multiviewPackageName in allMultiviewPackages:
            multiviewPackage = getattr(MultiviewClassifiers, multiviewPackageName)
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
                multiviewPackage = getattr(MultiviewClassifiers, multiviewPackageName)
                multiviewModule = getattr(multiviewPackage, multiviewPackageName+"Module")
                benchmark = multiviewModule.getBenchmark(benchmark, args=args)
    if "Monoview" in args.CL_type:
        if args.CL_algos_monoview == ['']:
            benchmark["Monoview"] = [name for _, name, isPackage in pkgutil.iter_modules(["./MonoMultiViewClassifiers/MonoviewClassifiers"])
                                     if not isPackage]

        else:
            benchmark["Monoview"] = args.CL_algos_monoview
    return benchmark


def genViewsDictionnary(DATASET):
    datasetsNames = DATASET.keys()
    viewsDictionary = dict((DATASET.get(datasetName).attrs["name"], int(datasetName[4:]))
                           for datasetName in datasetsNames
                           if datasetName[:4]=="View")
    return viewsDictionary


def initMonoviewExps(benchmark, argumentDictionaries, viewsDictionary,  NB_CLASS, kwargsInit):
    """Used to add each monoview exeperience args to the list of monoview experiences args"""
    if benchmark["Monoview"]:
        argumentDictionaries["Monoview"] = []
        for viewName, viewIndex in viewsDictionary.items():
            for classifier in benchmark["Monoview"]:
                arguments = {
                    "args": {classifier + "KWARGS": kwargsInit[classifier + "KWARGSInit"], "feat": viewName,
                             "CL_type": classifier, "nbClass": NB_CLASS}, "viewIndex": viewIndex}
                argumentDictionaries["Monoview"].append(arguments)
    return argumentDictionaries


def initMonoviewKWARGS(args, classifiersNames):
    """Used to init kwargs thanks to a function in each monoview classifier package"""
    monoviewKWARGS = {}
    for classifiersName in classifiersNames:
        classifierModule = getattr(MonoviewClassifiers, classifiersName)
        monoviewKWARGS[classifiersName + "KWARGSInit"] = classifierModule.getKWARGS(
            [(key, value) for key, value in vars(args).items() if key.startswith("CL_" + classifiersName)])
    return monoviewKWARGS


def initKWARGSFunc(args, benchmark):
    monoviewKWARGS = initMonoviewKWARGS(args, benchmark["Monoview"])
    return monoviewKWARGS


def initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries, randomState, directory,
                           resultsMonoview, classificationIndices):
    """Used to add each monoview exeperience args to the list of monoview experiences args"""
    multiviewArguments = []
    if "Multiview" in benchmark:
        for multiviewAlgoName in benchmark["Multiview"]:
            multiviewPackage = getattr(MultiviewClassifiers, multiviewAlgoName)
            mutliviewModule = getattr(multiviewPackage, multiviewAlgoName+"Module")
            multiviewArguments += mutliviewModule.getArgs(args, benchmark, views, viewsIndices, randomState, directory,
                                                          resultsMonoview, classificationIndices)
    argumentDictionaries["Multiview"] = multiviewArguments
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


def classifyOneIter_multicore(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices,
                              kFolds,
                              randomState, hyperParamSearch, metrics, coreIndex, viewsIndices, dataBaseTime, start,
                              benchmark,
                              views):
    """Used to execute mono and multiview classification and result analysis for one random state
     using multicore classification"""
    resultsMonoview = []
    labelsNames = LABELS_DICTIONARY.values()
    np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")

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
    logging.debug("Start:\t Analyze Iteration Results")
    resultAnalysis(benchmark, results, args.name, times, metrics, directory)
    logging.debug("Done:\t Analyze Iteration Results")
    globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
    totalTime = time.time() - start
    logging.info("Extraction time : " + str(int(dataBaseTime)) +
                 "s, Monoview time : " + str(int(monoviewTime)) +
                 "s, Multiview Time : " + str(int(multiviewTime)) +
                 "s, Iteration Analysis Time : " + str(int(globalAnalysisTime)) +
                 "s, Iteration Duration : " + str(int(totalTime)) + "s")
    return results, labelAnalysis


def classifyOneIter(LABELS_DICTIONARY, argumentDictionaries, nbCores, directory, args, classificationIndices, kFolds,
                    randomState, hyperParamSearch, metrics, DATASET, viewsIndices, dataBaseTime, start,
                    benchmark, views):
    """Used to execute mono and multiview classification and result analysis for one random state
         classification"""
    #TODO : Clarify this one


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
    logging.debug("Start:\t Analyze Iteration Results")
    resultAnalysis(benchmark, results, args.name, times, metrics, directory)
    logging.debug("Done:\t Analyze Iteration Results")
    globalAnalysisTime = time.time() - monoviewTime - dataBaseTime - start - multiviewTime
    totalTime = time.time() - start
    logging.info("Extraction time : " + str(int(dataBaseTime)) +
                 "s, Monoview time : " + str(int(monoviewTime)) +
                 "s, Multiview Time : " + str(int(multiviewTime)) +
                 "s, Iteration Analysis Time : " + str(int(globalAnalysisTime)) +
                 "s, Iteration Duration : " + str(int(totalTime)) + "s")
    return results, labelAnalysis


def getMetricsScoresBiclass(metrics, monoviewResults, multiviewResults):
    metricsScores = {}
    for metric in metrics:
        classifiersNames = []
        trainScores = []
        testScores = []
        for classifierResult in monoviewResults:
            trainScores.append(classifierResult[1][2][metric[0]][0])
            testScores.append(classifierResult[1][2][metric[0]][1])
            classifiersNames.append(classifierResult[1][0])
        for classifierResult in multiviewResults:
            trainScores.append(classifierResult[2][metric[0]][0])
            testScores.append(classifierResult[2][metric[0]][1])
            multiviewClassifierPackage = getattr(MultiviewClassifiers, classifierResult[0])
            multiviewClassifierModule = getattr(multiviewClassifierPackage, classifierResult[0]+"Module")
            classifiersNames.append(multiviewClassifierModule.genName(classifierResult[1]))
        metricsScores[metric[0]] = {"classifiersNames": classifiersNames,
                                    "trainScores": trainScores,
                                    "testScores": testScores}
    return metricsScores


def getExampleErrorsBiclass(usedBenchmarkArgumentDictionary, monoviewResults, multiviewResults):
    exampleErrors = {}
    trueLabels = usedBenchmarkArgumentDictionary["labels"]
    for classifierResult in monoviewResults:
        classifierName = classifierResult[1][0]
        predictedLabels = classifierResult[1][3]
        errorOnExamples = predictedLabels==trueLabels
        errorOnExamples = errorOnExamples.astype(int)
        unseenExamples = np.where(trueLabels==-100)[0]
        errorOnExamples[unseenExamples]=-100
        exampleErrors[classifierName] = errorOnExamples
    for classifierResult in multiviewResults:
        multiviewClassifierPackage = getattr(MultiviewClassifiers, classifierResult[0])
        multiviewClassifierModule = getattr(multiviewClassifierPackage, classifierResult[0]+"Module")
        classifierName = multiviewClassifierModule.genName(classifierResult[1])
        predictedLabels = classifierResult[3]
        errorOnExamples = predictedLabels==trueLabels
        errorOnExamples = errorOnExamples.astype(int)
        unseenExamples = np.where(trueLabels==-100)[0]
        errorOnExamples[unseenExamples]=-100
        exampleErrors[classifierName] = errorOnExamples
    return exampleErrors


def publishMetricsGraphs(metricsScores, directory, databaseName, labelsNames,minSize=10):
    for metricName, metricScores in metricsScores.items():
        logging.debug("Start:\t Multiclass score graph generation for "+metricName)
        trainScores = metricScores["trainScores"]
        testScores = metricScores["testScore"]
        names = metricScores["classifiersNames"]
        nbResults = len(testScores)

        testScores = np.array(testScores)
        trainScores = np.array(trainScores)
        names = np.array(names)
        size = nbResults
        if nbResults < minSize:
            size = minSize
        figKW = {"figsize" : (size, 3.0/4*size+2.0)}
        f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        barWidth= 0.35
        sorted_indices = np.argsort(testScores)
        testScores = testScores[sorted_indices]
        trainScores = trainScores[sorted_indices]
        names = names[sorted_indices]

        ax.set_title(metricName + "\n scores for each classifier")
        rects = ax.bar(range(nbResults), testScores, barWidth, color="r", )
        rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.7", )
        autolabel(rects, ax)
        autolabel(rect2, ax)
        ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
        ax.set_ylim(-0.1, 1.1)
        ax.set_xticks(np.arange(nbResults) + barWidth)
        ax.set_xticklabels(names, rotation="vertical")
        plt.tight_layout()
        f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + databaseName +"-"+"vs".join(labelsNames)+ "-" + metricName + ".png")
        plt.close()
        logging.debug("Done:\t Multiclass score graph generation for " + metricName)


def publishExampleErrors(exampleErrors, directory, databaseName, labelsNames,minSize=10):
    logging.debug("Start:\t Label analysis figure generation")
    nbClassifiers = len(exampleErrors)
    nbExamples = len(exampleErrors.values()[0])
    nbIter = 2
    data = np.zeros((nbExamples, nbClassifiers * nbIter))
    temp_data = np.zeros((nbExamples, nbClassifiers))
    classifiersNames = exampleErrors.keys()
    for classifierIndex, (classifierName, errorOnExamples) in enumerate(exampleErrors.items()):
        for iterIndex in range(nbIter):
            data[:, classifierIndex * nbIter + iterIndex] = errorOnExamples
            temp_data[:,classifierIndex] = errorOnExamples
    figWidth = max(nbClassifiers/2, minSize)
    figHeight = max(nbExamples/20, minSize)
    figKW = {"figsize":(figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    cmap = mpl.colors.ListedColormap(['black', 'red', 'green'])
    bounds = [-100.5,-0.5, 0.5, 1.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
    labels = classifiersNames
    plt.xticks(ticks, labels, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Unseen', 'Wrong', ' Right'])
    fig.tight_layout()
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + databaseName +"-"+"vs".join(labelsNames)+ "-error_analysis.png")
    plt.close()
    logging.debug("Done:\t Label analysis figure generation")

    logging.debug("Start:\t Error by example figure generation")
    errorOnExamples = -1*np.sum(data, axis=1)/nbIter+nbClassifiers
    np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors_doubled.csv", data, delimiter=",")
    np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", temp_data, delimiter=",")
    fig, ax = plt.subplots()
    x = np.arange(nbExamples)
    plt.bar(x, errorOnExamples)
    plt.ylim([0,nbClassifiers])
    plt.title("Number of classifiers that failed to classify each example")
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + databaseName +"-"+"vs".join(labelsNames)+ "-example_errors.png")
    plt.close()
    logging.debug("Done:\t Error by example figure generation")


def analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics):
    biclassResults = [{} for _ in range(statsIter)]
    for result in results:
        flag = result[0]
        iteridex = flag[0]
        classifierPositive = flag[1][0]
        classifierNegative = flag[1][1]
        biclassResults[iteridex][[classifierPositive, classifierNegative]] = {}
        for benchmarkArgumentDictionary in benchmarkArgumentDictionaries:
            if benchmarkArgumentDictionary["flag"]==flag:
                usedBenchmarkArgumentDictionary = benchmarkArgumentDictionary
        monoviewResults = result[1]
        multiviewResults = result[2]
        metricsScores = getMetricsScoresBiclass(metrics, monoviewResults, multiviewResults)
        exampleErrors = getExampleErrorsBiclass(usedBenchmarkArgumentDictionary, monoviewResults, multiviewResults)
        directory = usedBenchmarkArgumentDictionary["directory"]
        databaseName = usedBenchmarkArgumentDictionary["args"].name
        labelsNames = [usedBenchmarkArgumentDictionary["LABELS_DICTIONARY"][0],
                       usedBenchmarkArgumentDictionary["LABELS_DICTIONARY"][1]]
        publishMetricsGraphs(metricsScores, directory, databaseName, labelsNames)
        publishExampleErrors(exampleErrors, directory, databaseName, labelsNames)
        biclassResults[iteridex][[classifierPositive, classifierNegative]]["metricsScores"] = metricsScores
        biclassResults[iteridex][[classifierPositive, classifierNegative]]["exampleErrors"] = exampleErrors
    return biclassResults


def genMetricsScoresMulticlass(results, trueLabels, metrics, argumentsDictionaries, classificationIndices):
    """Used to add all the metrics scores to the multiclass result structure  for each clf and each iteration"""

    logging.debug("Start:\t Getting multiclass scores for each metric")
    # TODO : Metric score for train and test
    for metric in metrics:
        metricModule = getattr(Metrics, metric[0])
        for iterIndex, iterResults in enumerate(results):
            trainIndices, testIndices = classificationIndices[iterIndex]
            for classifierName, resultDictionary in iterResults.items():
                if not "metricsScores" in resultDictionary:
                    results[iterIndex][classifierName]["metricsScores"]={}
                trainScore = metricModule.score(trueLabels[trainIndices],
                                                resultDictionary["labels"][trainIndices])
                testScore = metricModule.score(trueLabels[testIndices],
                                                resultDictionary["labels"][testIndices])
                results[iterIndex][classifierName]["metricsScores"][metric[0]] = [trainScore, testScore]


    logging.debug("Done:\t Getting multiclass scores for each metric")

    return results


def getErrorOnLabelsMulticlass(multiclassResults, multiclassLabels):
    """Used to add all the arrays showing on which example there is an error for each clf and each iteration"""

    logging.debug("Start:\t Getting errors on each example for each classifier")

    for iterIndex, iterResults in enumerate(multiclassResults):
        for classifierName, classifierResults in iterResults.items():
            errorOnExamples = classifierResults["labels"] == multiclassLabels
            multiclassResults[iterIndex][classifierName]["errorOnExample"] = errorOnExamples.astype(int)

    logging.debug("Done:\t Getting errors on each example for each classifier")

    return multiclassResults


def autolabel(rects, ax):
    """Used to print scores on top of the bars"""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                "%.2f" % height,
                ha='center', va='bottom')


def publishMulticlassScores(multiclassResults, metrics, statsIter, direcories, databaseName, minSize=10):
    for iterIndex in range(statsIter):
        directory = direcories[iterIndex]
        for metric in metrics:
            logging.debug("Start:\t Multiclass score graph generation for "+metric[0])
            classifiersNames = []
            validationScores = []
            trainScores = []
            for classifierName in multiclassResults[iterIndex].keys():
                classifiersNames.append(classifierName)
                validationScores.append(multiclassResults[iterIndex][classifierName]["metricsScores"][metric[0]][1])
                trainScores.append(multiclassResults[iterIndex][classifierName]["metricsScores"][metric[0]][0])
            nbResults = len(validationScores)

            validationScores = np.array(validationScores)
            trainScores = np.array(trainScores)
            names = np.array(classifiersNames)
            size = nbResults
            if nbResults < minSize:
                size = minSize
            figKW = {"figsize" : (size, 3.0/4*size+2.0)}
            f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
            barWidth= 0.35
            sorted_indices = np.argsort(validationScores)
            validationScores = validationScores[sorted_indices]
            trainScores = trainScores[sorted_indices]
            names = names[sorted_indices]

            ax.set_title(metric[0] + "\n on validation set for each classifier")
            rects = ax.bar(range(nbResults), validationScores, barWidth, color="r", )
            rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.7", )
            autolabel(rects, ax)
            autolabel(rect2, ax)
            ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
            ax.set_ylim(-0.1, 1.1)
            ax.set_xticks(np.arange(nbResults) + barWidth)
            ax.set_xticklabels(names, rotation="vertical")
            plt.tight_layout()
            f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + databaseName + "-" + metric[0] + ".png")
            plt.close()
            logging.debug("Done:\t Multiclass score graph generation for " + metric[0])


def publishMulticlassExmapleErrors(multiclassResults, directories, databaseName, labelsNames, minSize=10):
    for iterIndex, multiclassResult in multiclassResults:
        directory = directories[iterIndex]
        logging.debug("Start:\t Label analysis figure generation")
        nbClassifiers = len(multiclassResult)
        nbExamples = len(multiclassResult.values()[0]["errorOnExample"])
        nbIter = 2
        data = np.zeros((nbExamples, nbClassifiers * nbIter))
        temp_data = np.zeros((nbExamples, nbClassifiers))
        classifiersNames = multiclassResult.keys()
        for classifierIndex, (classifierName, errorOnExamplesDict) in enumerate(multiclassResult.items()):
            for iterIndex in range(nbIter):
                data[:, classifierIndex * nbIter + iterIndex] = errorOnExamplesDict["errorOnExample"]
                temp_data[:,classifierIndex] = errorOnExamplesDict["errorOnExample"]
        figWidth = max(nbClassifiers/2, minSize)
        figHeight = max(nbExamples/20, minSize)
        figKW = {"figsize":(figWidth, figHeight)}
        fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        cmap = mpl.colors.ListedColormap(['black', 'red', 'green'])
        bounds = [-100.5,-0.5, 0.5, 1.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
        plt.title('Errors depending on the classifier')
        ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
        labels = classifiersNames
        plt.xticks(ticks, labels, rotation="vertical")
        cbar = fig.colorbar(cax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Unseen', 'Wrong', ' Right'])
        fig.tight_layout()
        fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + databaseName +"-error_analysis.png")
        plt.close()
        logging.debug("Done:\t Label analysis figure generation")

        logging.debug("Start:\t Error by example figure generation")
        errorOnExamples = -1*np.sum(data, axis=1)/nbIter+nbClassifiers
        np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors_doubled.csv", data, delimiter=",")
        np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", temp_data, delimiter=",")
        fig, ax = plt.subplots()
        x = np.arange(nbExamples)
        plt.bar(x, errorOnExamples)
        plt.ylim([0,nbClassifiers])
        plt.title("Number of classifiers that failed to classify each example")
        fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + databaseName +"-"+"vs".join(labelsNames)+ "-example_errors.png")
        plt.close()
        logging.debug("Done:\t Error by example figure generation")


def analyzeMulticlass(results, statsIter, benchmarkArgumentDictionaries, nbExamples, nbLabels, multiclassLabels,
                      metrics, classificationIndices, directories):
    """Used to tranform one versus one results in multiclass results and to publish it"""
    multiclassResults = [{} for _ in range(statsIter)]
    for flag, resMono, resMulti in results:
        iterIndex = flag[0]
        classifierPositive = flag[1][0]
        classifierNegative = flag[1][1]
        for classifierResult in resMono:
            if classifierResult[1][0] not in multiclassResults[iterIndex]:
                multiclassResults[iterIndex][classifierResult[1][0]] = np.zeros((nbExamples,nbLabels),dtype=int)
            for exampleIndex, label in enumerate(classifierResult[1][3]):
                if label == 1:
                    multiclassResults[iterIndex][classifierResult[1][0]][exampleIndex, classifierPositive] += 1
                else:
                    multiclassResults[iterIndex][classifierResult[1][0]][exampleIndex, classifierNegative] += 1
        for classifierResult in resMulti:
            multiviewClassifierPackage = getattr(MultiviewClassifiers, classifierResult[0])
            multiviewClassifierModule = getattr(multiviewClassifierPackage, classifierResult[0]+"Module")
            classifierName = multiviewClassifierModule.genName(classifierResult[1])
            if classifierName not in multiclassResults[iterIndex]:
                multiclassResults[iterIndex][classifierName] = np.zeros((nbExamples,nbLabels),dtype=int)
            for exampleIndex, label in enumerate(classifierResult[3]):
                if label == 1:
                    multiclassResults[iterIndex][classifierName][exampleIndex, classifierPositive] += 1
                else:
                    multiclassResults[iterIndex][classifierName][exampleIndex, classifierNegative] += 1


    for iterIndex, multiclassiterResult in enumerate(multiclassResults):
        for key, value in multiclassiterResult.items():
            multiclassResults[iterIndex][key] = {"labels": np.argmax(value, axis=1)}

    multiclassResults = genMetricsScoresMulticlass(multiclassResults, multiclassLabels, metrics, benchmarkArgumentDictionaries, classificationIndices)
    multiclassResults = getErrorOnLabelsMulticlass(multiclassResults, multiclassLabels)

    publishMulticlassScores(multiclassResults, metrics, statsIter, directories, benchmarkArgumentDictionaries[0]["args"].name)
    publishMulticlassExmapleErrors(multiclassResults, metrics, statsIter, directories, benchmarkArgumentDictionaries[0]["args"].name)
    return multiclassResults


def publishIterBiclassMetricsScores(iterResults, directory, labelsDictionary, classifiersDict, dataBaseName, statsIter, minSize=10):
    for labelsCombination, iterResult in iterResults.items():
        currentDirectory = directory+ labelsDictionary[labelsCombination[0]]+"vs"+labelsDictionary[labelsCombination[1]]+"/"
        for metricName, scores in iterResults["metricsScores"].items():
            trainScores = scores["trainScores"]
            testScores = scores["testScores"]
            trainMeans = np.mean(trainScores, axis=1)
            testMeans = np.mean(testScores, axis=1)
            trainSTDs = np.std(trainScores, axis=1)
            testSTDs = np.std(testScores, axis=1)
            nbResults = len(trainMeans)
            names = classifiersDict.values()
            size=nbResults
            if nbResults<minSize:
                size=minSize
            figKW = {"figsize" : (size, 3.0/4*size+2.0)}
            f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
            barWidth = 0.35  # the width of the bars
            sorted_indices = np.argsort(testMeans)
            testMeans = testMeans[sorted_indices]
            testSTDs = testSTDs[sorted_indices]
            trainSTDs = trainSTDs[sorted_indices]
            trainMeans = trainMeans[sorted_indices]
            names = np.array(names)[sorted_indices]

            ax.set_title(metricName + " for each classifier")
            rects = ax.bar(range(nbResults), testMeans, barWidth, color="r", yerr=testSTDs)
            rect2 = ax.bar(np.arange(nbResults) + barWidth, trainMeans, barWidth, color="0.7", yerr=trainSTDs)
            autolabel(rects, ax)
            autolabel(rect2, ax)
            ax.set_ylim(-0.1, 1.1)
            ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
            ax.set_xticks(np.arange(nbResults) + barWidth)
            ax.set_xticklabels(names, rotation="vertical")
            f.tight_layout()
            f.savefig(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-" + dataBaseName + "-Mean_on_"
                      + str(statsIter) + "_iter-" + metricName + ".png")



def publishIterBiclassExampleErrors(iterResults, directory, labelsDictionary, classifiersDict, statsIter, minSize=10):
    for labelsCombination, combiResults in iterResults.items():
        currentDirectory = directory+ labelsDictionary[labelsCombination[0]]+"vs"+labelsDictionary[labelsCombination[1]]+"/"
        classifiersNames = classifiersDict.values()
        logging.debug("Start:\t Global label analysis figure generation")
        nbExamples = combiResults["errorOnExamples"].shape[1]
        nbClassifiers = combiResults["errorOnExamples"].shape[0]
        nbIter = 2

        figWidth = max(nbClassifiers / 2, minSize)
        figHeight = max(nbExamples / 20, minSize)
        figKW = {"figsize": (figWidth, figHeight)}
        fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        data = combiResults["errorOnExamples"]
        cax = plt.imshow(-data, interpolation='none', cmap="Greys", aspect='auto')
        plt.title('Errors depending on the classifier')
        ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
        plt.xticks(ticks, classifiersNames, rotation="vertical")
        cbar = fig.colorbar(cax, ticks=[0, -statsIter])
        cbar.ax.set_yticklabels(['Always Wrong', 'Always Right'])
        fig.tight_layout()
        fig.savefig(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
        plt.close()
        logging.debug("Done:\t Global label analysis figure generation")
        logging.debug("Start:\t Global error by example figure generation")
        errorOnExamples = -1 * np.sum(data, axis=1) / nbIter + (nbClassifiers*statsIter)
        np.savetxt(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors.csv", data, delimiter=",")
        np.savetxt(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", errorOnExamples, delimiter=",")
        fig, ax = plt.subplots()
        x = np.arange(nbExamples)
        plt.bar(x, errorOnExamples)
        plt.ylim([0,nbClassifiers*statsIter])
        plt.title("Number of classifiers that failed to classify each example")
        fig.savefig(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.png")
        plt.close()
        logging.debug("Done:\t Global error by example figure generation")


def publishIterMulticlassMetricsScores(iterMulticlassResults, classifiersNames, dataBaseName, directory, statsIter, minSize=10):
    for metricName, scores in iterMulticlassResults["metricsScores"].items():
        trainScores = scores["trainScores"]
        testScores = scores["testScores"]
        trainMeans = np.mean(trainScores, axis=1)
        testMeans = np.mean(testScores, axis=1)
        trainSTDs = np.std(trainScores, axis=1)
        testSTDs = np.std(testScores, axis=1)
        nbResults = len(trainMeans)
        names = classifiersNames
        size=nbResults
        if nbResults<minSize:
            size=minSize
        figKW = {"figsize" : (size, 3.0/4*size+2.0)}
        f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        barWidth = 0.35  # the width of the bars
        sorted_indices = np.argsort(testMeans)
        testMeans = testMeans[sorted_indices]
        testSTDs = testSTDs[sorted_indices]
        trainSTDs = trainSTDs[sorted_indices]
        trainMeans = trainMeans[sorted_indices]
        names = np.array(names)[sorted_indices]

        ax.set_title(metricName + " for each classifier")
        rects = ax.bar(range(nbResults), testMeans, barWidth, color="r", yerr=testSTDs)
        rect2 = ax.bar(np.arange(nbResults) + barWidth, trainMeans, barWidth, color="0.7", yerr=trainSTDs)
        autolabel(rects, ax)
        autolabel(rect2, ax)
        ax.set_ylim(-0.1, 1.1)
        ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
        ax.set_xticks(np.arange(nbResults) + barWidth)
        ax.set_xticklabels(names, rotation="vertical")
        f.tight_layout()
        f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + dataBaseName + "-Mean_on_"
                  + str(statsIter) + "_iter-" + metricName + ".png")


def publishIterMulticlassExampleErrors(iterMulticlassResults, directory, classifiersNames, statsIter, minSize=10):
    logging.debug("Start:\t Global label analysis figure generation")
    nbExamples = iterMulticlassResults["errorOnExamples"].shape[1]
    nbClassifiers = iterMulticlassResults["errorOnExamples"].shape[0]
    nbIter = 2

    figWidth = max(nbClassifiers / 2, minSize)
    figHeight = max(nbExamples / 20, minSize)
    figKW = {"figsize": (figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    data = iterMulticlassResults["errorOnExamples"]
    cax = plt.imshow(-data, interpolation='none', cmap="Greys", aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
    plt.xticks(ticks, classifiersNames, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[0, -statsIter])
    cbar.ax.set_yticklabels(['Always Wrong', 'Always Right'])
    fig.tight_layout()
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
    plt.close()
    logging.debug("Done:\t Global label analysis figure generation")
    logging.debug("Start:\t Global error by example figure generation")
    errorOnExamples = -1 * np.sum(data, axis=1) / nbIter + (nbClassifiers*statsIter)
    np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors.csv", data, delimiter=",")
    np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", errorOnExamples, delimiter=",")
    fig, ax = plt.subplots()
    x = np.arange(nbExamples)
    plt.bar(x, errorOnExamples)
    plt.ylim([0,nbClassifiers*statsIter])
    plt.title("Number of classifiers that failed to classify each example")
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.png")
    plt.close()
    logging.debug("Done:\t Global error by example figure generation")


def analyzeIterMulticlass(biclassResults, multiclassResults, directory, statsIter, labelsDictionary, metrics, dataBaseName):
    iterBiclassResults = {}
    classifiersDict = {}

    for iterIndex, biclassResult in biclassResults:
        for labelsComination, results in biclassResult.items():
            for metric in metrics:
                nbClassifiers = len(results["metricsScores"][metric[0]]["classifiersNames"])
                if not classifiersDict:
                    classifiersDict = dict((classifierName, classifierIndex)
                                           for classifierIndex, classifierName
                                           in enumerate(results["metricsScores"][metric[0]]["classifiersNames"]))
                if labelsComination not in iterBiclassResults:
                    iterBiclassResults[labelsComination] = {}
                    iterBiclassResults[labelsComination]["metricsScores"] = {}

                    iterBiclassResults[labelsComination]["errorOnExamples"] = np.zeros((nbClassifiers,
                                                                                        len(results["exampleErrors"])),
                                                                                       dtype=int)
                if metric[0] not in iterBiclassResults[labelsComination]["metricsScores"]:
                    iterBiclassResults[labelsComination]["metricsScores"][metric[0]]= {"trainScores":
                                                                                           np.zeros(nbClassifiers, statsIter),
                                                                                       "testScores":
                                                                                           np.zeros(nbClassifiers, statsIter)}
                for classifierName, trainScore, testScore in zip(results["metricsScores"][metric[0]]["classifiersNames"],
                                                                 results["metricsScores"][metric[0]]["trainScores"],
                                                                 results["metricsScores"][metric[0]]["testScores"],
                                                                 ):
                    iterBiclassResults[labelsComination]["metricsScores"][metric[0]]["trainScores"][classifiersDict[classifierName], iterIndex] = trainScore
                    iterBiclassResults[labelsComination]["metricsScores"][metric[0]]["testScores"][classifiersDict[classifierName], iterIndex] = testScore
            for classifierName, errorOnExample in results["errorOnExamples"]:
                iterBiclassResults[labelsComination]["errorOnExamples"][classifiersDict[classifierName], :] += errorOnExample
    publishIterBiclassMetricsScores(iterBiclassResults, directory, labelsDictionary, classifiersDict, dataBaseName, statsIter)
    publishIterBiclassExampleErrors(iterBiclassResults, directory, labelsDictionary, classifiersDict, statsIter)

    iterMulticlassResults = {}
    nbClassifiers = len(multiclassResults[0])
    iterMulticlassResults["errorOnExamples"] = np.zeros((nbClassifiers,len(multiclassResults[0].values()[0]["exampleErrors"])),
                                                         dtype=int)
    iterMulticlassResults["metricsScores"] = {}
    classifiersNames = []
    for iterIndex, multiclassResult in multiclassResults:
        for classifierName, classifierResults in multiclassResult.items():
            classifiersNames.append(classifierName)
            classifierIndex = len(classifiersNames)-1
            for metric in metrics:
                if metric[0] not in iterMulticlassResults["metricsScores"]:
                    iterMulticlassResults["metricsScores"][metric[0]] = {"trainScores":
                                                                             np.zeros(nbClassifiers, statsIter),
                                                                         "testScores":
                                                                             np.zeros(nbClassifiers, statsIter)}
                iterMulticlassResults["metricsScores"][metric[0]]["trainScores"][classifierIndex, iterIndex] = classifierResults["metricsScores"][metric[0]][0]
                iterMulticlassResults["metricsScores"][metric[0]]["testScores"][classifierIndex, iterIndex] = classifierResults["metricsScores"][metric[0]][1]
            iterMulticlassResults["errorOnExamples"][classifierIndex, :] = classifierResults["exampleErrors"]
    publishIterMulticlassMetricsScores(iterMulticlassResults, classifiersNames, dataBaseName, directory, statsIter)
    publishIterMulticlassExampleErrors(iterMulticlassResults, directory, classifiersNames, statsIter)



def getResults(results, statsIter, nbMulticlass, benchmarkArgumentDictionaries, multiclassLabels, metrics, classificationIndices, directories, directory):
    if statsIter > 1:
        if nbMulticlass > 1:
            biclassResults = analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics)
            multiclassResults = analyzeMulticlass(results, statsIter, benchmarkArgumentDictionaries, multiclassLabels, metrics, classificationIndices, directories)
            datatBaseName = benchmarkArgumentDictionaries[0]["args"].name
            analyzeIterMulticlass(biclassResults, multiclassResults, directory, statsIter, dataBaseName)
        else:
            biclassResults = analyzeBiclass(results)
            analyzeBiclassIter(biclassResults)
    else:
        if nbMulticlass>1:
            analyzeMulticlass(results)
        else:
            analyzeBiclass(results)


def execOneBenchmark(coreIndex=-1, LABELS_DICTIONARY=None, directory=None, classificationIndices=None, args=None,
                     kFolds=None, randomState=None, hyperParamSearch=None, metrics=None, argumentDictionaries=None,
                     benchmark=None, views=None, viewsIndices=None, flag=None, labels=None,
                     ExecMonoview_multicore=ExecMonoview_multicore, ExecMultiview_multicore=ExecMultiview_multicore,
                     initMultiviewArguments=initMultiviewArguments):
    """Used to run a benchmark using one core. ExecMonoview_multicore, initMultiviewArguments and
     ExecMultiview_multicore args are only used for tests"""
    if not os.path.exists(os.path.dirname(directory + "train_labels.csv")):
        try:
            os.makedirs(os.path.dirname(directory + "train_labels.csv"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    trainIndices, testIndices = classificationIndices
    trainLabels = labels[trainIndices]
    np.savetxt(directory + "train_labels.csv", trainLabels, delimiter=",")
    resultsMonoview = []
    labelsNames = list(LABELS_DICTIONARY.values())
    np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")
    resultsMonoview += [ExecMonoview_multicore(directory, args.name, labelsNames, classificationIndices, kFolds,
                                               coreIndex, args.type, args.pathF, randomState, labels,
                                               hyperParamSearch=hyperParamSearch, metrics=metrics,
                                               nIter=args.CL_GS_iter, **argument)
                        for argument in argumentDictionaries["Monoview"]]

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)

    resultsMultiview = []
    resultsMultiview += [
        ExecMultiview_multicore(directory, coreIndex, args.name, classificationIndices, kFolds, args.type,
                                args.pathF, LABELS_DICTIONARY, randomState, labels, hyperParamSearch=hyperParamSearch,
                                metrics=metrics, nIter=args.CL_GS_iter, **arguments)
        for arguments in argumentDictionaries["Multiview"]]
    return [flag, resultsMonoview, resultsMultiview]


def execOneBenchmark_multicore(nbCores=-1, LABELS_DICTIONARY=None, directory=None, classificationIndices=None, args=None,
                               kFolds=None, randomState=None, hyperParamSearch=None, metrics=None, argumentDictionaries=None,
                               benchmark=None, views=None, viewsIndices=None, flag=None, labels=None,
                               ExecMonoview_multicore=ExecMonoview_multicore,
                               ExecMultiview_multicore=ExecMultiview_multicore,
                               initMultiviewArguments=initMultiviewArguments):
    """Used to run a benchmark using multiple cores. ExecMonoview_multicore, initMultiviewArguments and
     ExecMultiview_multicore args are only used for tests"""
    if not os.path.exists(os.path.dirname(directory + "train_labels.csv")):
        try:
            os.makedirs(os.path.dirname(directory + "train_labels.csv"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    trainIndices, testIndices = classificationIndices
    trainLabels = labels[trainIndices]
    np.savetxt(directory + "train_labels.csv", trainLabels, delimiter=",")
    np.savetxt(directory + "train_indices.csv", classificationIndices[0], delimiter=",")
    resultsMonoview = []
    labelsNames = list(LABELS_DICTIONARY.values())

    nbExperiments = len(argumentDictionaries["Monoview"])
    nbMulticoreToDo = int(math.ceil(float(nbExperiments) / nbCores))
    for stepIndex in range(nbMulticoreToDo):
        resultsMonoview += (Parallel(n_jobs=nbCores)(
            delayed(ExecMonoview_multicore)(directory, args.name, labelsNames, classificationIndices, kFolds,
                                            coreIndex, args.type, args.pathF, randomState, labels,
                                            hyperParamSearch=hyperParamSearch,
                                            metrics=metrics, nIter=args.CL_GS_iter,
                                            **argumentDictionaries["Monoview"][coreIndex + stepIndex * nbCores])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores))))

    argumentDictionaries = initMultiviewArguments(args, benchmark, views, viewsIndices, argumentDictionaries,
                                                  randomState, directory, resultsMonoview, classificationIndices)

    resultsMultiview = []
    nbExperiments = len(argumentDictionaries["Multiview"])
    nbMulticoreToDo = int(math.ceil(float(nbExperiments) / nbCores))
    for stepIndex in range(nbMulticoreToDo):
        resultsMultiview += Parallel(n_jobs=nbCores)(
            delayed(ExecMultiview_multicore)(directory, coreIndex, args.name, classificationIndices, kFolds,
                                             args.type, args.pathF, LABELS_DICTIONARY, randomState, labels,
                                             hyperParamSearch=hyperParamSearch, metrics=metrics, nIter=args.CL_GS_iter,
                                             **argumentDictionaries["Multiview"][stepIndex * nbCores + coreIndex])
            for coreIndex in range(min(nbCores, nbExperiments - stepIndex * nbCores)))

    return [flag, resultsMonoview, resultsMultiview]


def execOneBenchmarkMonoCore(coreIndex=-1, LABELS_DICTIONARY=None, directory=None, classificationIndices=None, args=None,
                             kFolds=None, randomState=None, hyperParamSearch=None, metrics=None, argumentDictionaries=None,
                             benchmark=None, views=None, viewsIndices=None, flag=None, labels=None,
                             ExecMonoview_multicore=ExecMonoview_multicore, ExecMultiview_multicore=ExecMultiview_multicore,
                             initMultiviewArguments=initMultiviewArguments):
    pass


def execBenchmark(nbCores, statsIter, nbMulticlass, benchmarkArgumentsDictionaries, classificationIndices, directories, directory,
                  execOneBenchmark=execOneBenchmark, execOneBenchmark_multicore=execOneBenchmark_multicore,
                  execOneBenchmarkMonoCore=execOneBenchmarkMonoCore):
    """Used to execute the needed benchmark(s) on multicore or mono-core functions
    The execOneBenchmark and execOneBenchmark_multicore keywords args are only used in the tests"""
    # TODO :  find a way to flag

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
            results += [execOneBenchmarkMonoCore(**arguments)]
    logging.debug("Done:\t Executing all the needed biclass benchmarks")

    # Do everything with flagging

    logging.debug("Start:\t Analyzing preds")
    # getResults(results, statsIter, nbMulticlass, benchmarkArgumentsDictionaries, multiclassLabels, metrics, classificationIndices, directories, directory)
    logging.debug("Done:\t Analyzing preds")

    return results


def execClassif(arguments):
    """Main function to execute the benchmark"""
    start = time.time()
    args = execution.parseTheArgs(arguments)

    os.nice(args.nice)
    nbCores = args.nbCores
    statsIter = args.CL_statsiter
    hyperParamSearch = args.CL_HPS_type
    multiclassMethod = args.CL_multiclassMethod

    directory = execution.initLogFile(args)
    randomState = execution.initRandomState(args.randomState, directory)
    if statsIter > 1:
        statsIterRandomStates = [np.random.RandomState(randomState.randint(500)) for _ in range(statsIter)]
    else:
        statsIterRandomStates = [randomState]

    if args.name not in ["Fake", "Plausible"]:
        getDatabase = getattr(DB, "getClassicDB" + args.type[1:])
    else:
        getDatabase = getattr(DB, "get" + args.name + "DB" + args.type[1:])

    DATASET, LABELS_DICTIONARY = getDatabase(args.views, args.pathF, args.name, args.CL_nbClass,
                                             args.CL_classes)

    classificationIndices = execution.genSplits(DATASET.get("Labels").value, args.CL_split, statsIterRandomStates)

    multiclassLabels, labelsCombinations, oldIndicesMulticlass = Multiclass.genMulticlassLabels(DATASET.get("Labels").value, multiclassMethod, classificationIndices)

    kFolds = execution.genKFolds(statsIter, args.CL_nbFolds, statsIterRandomStates)

    datasetFiles = Dataset.initMultipleDatasets(args, nbCores)

    # views, viewsIndices, allViews = execution.initViews(DATASET, args)
    # if not views:
    #     raise ValueError("Empty views list, modify selected views to match dataset " + args.views)
    viewsDictionary = genViewsDictionnary(DATASET)

    # NB_VIEW = DATASET.get("Metadata").attrs["nbViews"]
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

    logging.debug("Start:\t Finding all available mono- & multiview algorithms")

    benchmark = initBenchmark(args)

    initKWARGS = initKWARGSFunc(args, benchmark)

    dataBaseTime = time.time() - start

    argumentDictionaries = {"Monoview": [], "Multiview": []}
    argumentDictionaries = initMonoviewExps(benchmark, argumentDictionaries, viewsDictionary, NB_CLASS,
                                            initKWARGS)
    directories = execution.genDirecortiesNames(directory, statsIter)
    benchmarkArgumentDictionaries = execution.genArgumentDictionaries(LABELS_DICTIONARY, directories, multiclassLabels,
                                                                      labelsCombinations, oldIndicesMulticlass,
                                                                      hyperParamSearch, args, kFolds,
                                                                      statsIterRandomStates, metrics,
                                                                      argumentDictionaries, benchmark)

    nbMulticlass = len(labelsCombinations)

    execBenchmark(nbCores, statsIter, nbMulticlass, benchmarkArgumentDictionaries, classificationIndices, directories, directory)



















    if statsIter > 1:
        logging.debug("Start:\t Benchmark classification")
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
            nbExperiments = statsIter*len(multiclassLabels)
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
        logging.debug("Done:\t Benchmark classification")
        logging.debug("Start:\t Global Results Analysis")
        classifiersIterResults = []
        iterLabelAnalysis = []
        for result in iterResults:
            classifiersIterResults.append(result[0])
            iterLabelAnalysis.append(result[1])

        mono,multi = classifiersIterResults[0]
        classifiersNames = genNamesFromRes(mono, multi)
        analyzeIterLabels(iterLabelAnalysis, directory, classifiersNames)
        analyzeIterResults(classifiersIterResults, args.name, metrics, directory)
        logging.debug("Done:\t Global Results Analysis")
        totalDur = time.time() - start
        m, s = divmod(totalDur, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        # print "%d:%02d:%02d" % (h, m, s)
        logging.info("Info:\t Total duration : " + str(d) + " days, " + str(h) + " hours, " + str(m) + " mins, " + str(
            int(s)) + "secs.")

    else:
        logging.debug("Start:\t Benchmark classification")
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
        logging.debug("Done:\t Benchmark classification")
        totalDur = time.time()-start
        m, s = divmod(totalDur, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        # print "%d:%02d:%02d" % (h, m, s)
        logging.info("Info:\t Total duration : "+str(d)+ " days, "+str(h)+" hours, "+str(m)+" mins, "+str(int(s))+"secs.")

    if statsIter > 1:
        pass
