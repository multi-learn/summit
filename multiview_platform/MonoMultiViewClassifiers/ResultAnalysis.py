# Import built-in modules
import time
import os
import pylab
import errno
import logging

# Import third party modules
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
# from matplotlib import cm
import matplotlib as mpl

# Import own Modules
from . import Metrics
from . import MultiviewClassifiers

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def autolabel(rects, ax, set=1, std=None):
    r"""Used to print the score below the bars.

    Parameters
    ----------
    rects : pyplot bar object
        THe bars.
    ax : pyplot ax object
        The ax.
    set : integer
        1 means the test scores, anything else means the train score
    std: None or array
        The standard deviations in the case of statsIter results.

    Returns
    -------
    """
    if set == 1:
        text_height = -0.05
        weight = "bold"
    else:
        text_height = -0.07
        weight = "normal"
    for rectIndex, rect in enumerate(rects):
        height = rect.get_height()
        if std is not None:
            ax.text(rect.get_x() + rect.get_width() / 2., text_height,
                    "%.2f" % height + u'\u00B1' + "%.2f" % std[rectIndex], weight=weight,
                    ha='center', va='bottom', size="x-small")
        else:
            ax.text(rect.get_x() + rect.get_width() / 2., text_height,
                    "%.2f" % height, weight=weight,
                    ha='center', va='bottom', size="small")


def getMetricsScoresBiclass(metrics, results):
    r"""Used to extract metrics scores in case of biclass classification

    Parameters
    ----------
    metrics : list of lists
        The metrics names with configuration metrics[i][0] = name of metric i
    results : list of MonoviewResult and MultiviewResults objects
        A list containing all the resluts for all the monoview experimentations.

    Returns
    -------
    metricsScores : dict of dict of list
    Regroups all the scores for each metrics for each classifier and for the train and test sets.
    organized as :
    -`metricScores[metric_name]["classifiersNames"]` is a list of all the classifiers available for this metric,
    -`metricScores[metric_name]["trainScores"]` is a list of all the available classifiers scores on the train set,
    -`metricScores[metric_name]["testScores"]` is a list of all the available classifiers scores on the test set.
    """
    metricsScores = {}

    for metric in metrics:
        classifiersNames = []
        trainScores = []
        testScores = []

        for classifierResult in results:
            trainScores.append(classifierResult.metrics_scores[metric[0]][0])
            testScores.append(classifierResult.metrics_scores[metric[0]][1])
            classifiersNames.append(classifierResult.get_classifier_name())

        metricsScores[metric[0]] = {"classifiersNames": classifiersNames,
                                    "trainScores": trainScores,
                                    "testScores": testScores}
    return metricsScores


def getExampleErrorsBiclass(groud_truth, results):
    r"""Used to get for each classifier and each example whether the classifier has misclassified the example or not.

    Parameters
    ----------
    ground_truth : numpy array of 0, 1 and -100 (if multiclass)
        The array with the real labels of the examples
    results : list of MonoviewResult and MultiviewResults objects
        A list containing all the resluts for all the mono- & multi-view experimentations.

    Returns
    -------
    exampleErrors : dict of np.array
        For each classifier, has an entry with a `np.array` over the examples, with a 1 if the examples was
        well-classified, a 0 if not and if it's multiclass classification, a -100 if the examples was not seen during
        the one versus one classification.
    """
    exampleErrors = {}

    for classifierResult in results:
        errorOnExamples = np.equal(classifierResult.full_labels_pred, groud_truth).astype(int)
        unseenExamples = np.where(groud_truth==-100)[0]
        errorOnExamples[unseenExamples]=-100
        exampleErrors[classifierResult.get_classifier_name()] = errorOnExamples

    return exampleErrors


def get_fig_size(nb_results, min_size=15, multiplier=1.0, bar_width=0.35):
    r"""Used to get the image size to save the figure and the bar width, depending on the number of scores to plot.

    Parameters
    ----------
    nb_results : int
        The number of couple of bar to plot.
    min_size : int
        The minimum size of the image, if there are few classifiers to plot.
    multiplier : float
        The ratio between the image size and the number of classifiers.
    bar_width : float
        The width of the bars in the figure. Mainly here to centralize bar_width.

    Returns
    -------
    fig_kwargs : dict of arguments
        The argument restraining the size of the figure, usable directly in the `subplots` function of
        `matplotlib.pyplot`.
    bar_width : float
        The width of the bars in the figure. Mainly here to centralize bar_width.
    """
    size = nb_results*multiplier
    if size < min_size:
        size = min_size
    fig_kwargs = {"figsize": (size, size / 3)}
    return fig_kwargs, bar_width


def sort_by_test_score(train_scores,test_scores, names, train_STDs=None, test_STDs=None):
    r"""Used to sort the results (names and both scores) in descending test score order.

    Parameters
    ----------
    train_scores : np.array of floats
        The scores of each classifier on the training set.
    test_scores : np.array of floats
        The scores of each classifier on the testing set.
    names : np.array of strs
        The names of all the classifiers.
    train_STDs : np.array of floats or None
        The array containing the standard deviations for the averaged scores on the training set.
    test_STDs : np.array of floats or None
        The array containing the standard deviations for the averaged scores on the testing set.

    Returns
    -------
    sorted_names : np.array of strs
        The names of all the classifiers, sorted in descending test score order.
    sorted_train_scores : np.array of floats
        The scores of each classifier on the training set, sorted in descending test score order.
    sorted_test_scores : np.array of floats
        The scores of each classifier on the testing set, sorted in descending test score order.
    sorted_train_STDs : np.array of floats or None
        The array containing the standard deviations for the averaged scores on the training set,
        sorted in descending test score order.
    sorted_test_STDs : np.array of floats or None
        The array containing the standard deviations for the averaged scores on the testing set,
        sorted in descending test score order.
    """
    sorted_indices = np.argsort(test_scores)
    sorted_test_scores = test_scores[sorted_indices]
    sorted_train_scores = train_scores[sorted_indices]
    sorted_names = names[sorted_indices]
    if train_STDs is not None and test_STDs is not None:
        sorted_train_STDs = train_STDs[sorted_indices]
        sorted_test_STDs = test_STDs[sorted_indices]
    else:
        sorted_train_STDs = None
        sorted_test_STDs = None
    return sorted_names, sorted_train_scores, sorted_test_scores, sorted_train_STDs, sorted_test_STDs


def plotMetricScores(trainScores, testScores, names, nbResults, metricName, fileName,
                     tag="", train_STDs=None, test_STDs=None):
    r"""Used to plot and save the score barplot for a specific metric.

    Parameters
    ----------
    trainScores : list or np.array of floats
        The scores of each classifier on the training set.
    testScores : list or np.array of floats
        The scores of each classifier on the testing set.
    names : list or np.array of strs
        The names of all the classifiers.
    nbResults: int
        The number of classifiers to plot.
    metricName : str
        The plotted metric's name
    fileName : str
        The name of the file where the figure will be saved.

    Returns
    -------
    """

    figKW, barWidth = get_fig_size(nbResults)

    names, trainScores, testScores, train_STDs, test_STDs = sort_by_test_score(trainScores, testScores, names,
                                                                               train_STDs, test_STDs)

    f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    ax.set_title(metricName + "\n"+ tag +" scores for each classifier")

    rects = ax.bar(range(nbResults), testScores, barWidth, color="0.8", yerr=test_STDs)
    rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.5", yerr=train_STDs)
    autolabel(rects, ax, set=1, std=test_STDs)
    autolabel(rect2, ax, set=2, std=train_STDs)

    ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(nbResults) + barWidth)
    ax.set_xticklabels(names, rotation="vertical")

    plt.tight_layout()
    f.savefig(fileName)
    plt.close()


def publishMetricsGraphs(metricsScores, directory, databaseName, labelsNames):
    r"""Used to sort the results (names and both scores) in descending test score order.

    Parameters
    ----------
    metricsScores : dict of dicts of lists or np.arrays
        Keys : The names of the metrics.
        Values : The scores and names of each classifier .
    directory : str
        The path to the directory where the figures will be saved.
    databaseName : str
        The name of the database on which the experiments where conducted.
    labelsNames : list of strs
        The name corresponding to each numerical label.

    Returns
    -------
    """
    for metricName, metricScores in metricsScores.items():
        logging.debug("Start:\t Biclass score graph generation for "+metricName)

        nbResults = len(metricScores["testScores"])

        fileName = directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + databaseName +"-"+"_vs_".join(labelsNames)+ "-" + metricName + ".png"

        plotMetricScores(np.array(metricScores["trainScores"]), np.array(metricScores["testScores"]),
                         np.array(metricScores["classifiersNames"]), nbResults, metricName, fileName, tag=" "+" vs ".join(labelsNames))

        logging.debug("Done:\t Biclass score graph generation for " + metricName)


def publishExampleErrors(exampleErrors, directory, databaseName, labelsNames, minSize=10):
    logging.debug("Start:\t Biclass Label analysis figure generation")
    nbClassifiers = len(exampleErrors)
    nbExamples = len(list(exampleErrors.values())[0])
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

    plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
    labels = classifiersNames
    plt.xticks(ticks, labels, rotation="vertical")
    red_patch = mpatches.Patch(color='red', label='Classifier failed')
    green_patch = mpatches.Patch(color='green', label='Classifier succeded')
    black_patch = mpatches.Patch(color='black', label='Unseen data')
    plt.legend(handles=[red_patch, green_patch, black_patch],
               bbox_to_anchor=(0,1.02,1,0.2),
               loc="lower left",
               mode="expand",
               borderaxespad=0,
               ncol=3)
    fig.tight_layout()
    fig.savefig(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + databaseName +"-"+"_vs_".join(labelsNames)+ "-error_analysis.png", bbox_inches="tight")
    plt.close()
    logging.debug("Done:\t Biclass Label analysis figure generation")

    logging.debug("Start:\t Biclass Error by example figure generation")
    errorOnExamples = -1*np.sum(data, axis=1)/nbIter+nbClassifiers
    np.savetxt(directory + "clf_errors_doubled.csv", data, delimiter=",")
    np.savetxt(directory + "example_errors.csv", temp_data, delimiter=",")
    fig, ax = plt.subplots()
    x = np.arange(nbExamples)
    plt.bar(x, errorOnExamples)
    plt.ylim([0,nbClassifiers])
    plt.title("Number of classifiers that failed to classify each example")
    fig.savefig(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + databaseName +"-"+"_vs_".join(labelsNames)+ "-example_errors.png")
    plt.close()
    logging.debug("Done:\t Biclass Error by example figure generation")


def analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics):
    logging.debug("Srart:\t Analzing all biclass resuls")
    biclassResults = [{} for _ in range(statsIter)]
    for flag, result in results:
        iteridex = flag[0]
        classifierPositive = flag[1][0]
        classifierNegative = flag[1][1]
        biclassResults[iteridex][str(classifierPositive) + str(classifierNegative)] = {}

        for benchmarkArgumentDictionary in benchmarkArgumentDictionaries:
            if benchmarkArgumentDictionary["flag"]==flag:
                usedBenchmarkArgumentDictionary = benchmarkArgumentDictionary
        metricsScores = getMetricsScoresBiclass(metrics, result)
        exampleErrors = getExampleErrorsBiclass(usedBenchmarkArgumentDictionary["labels"], result)
        directory = usedBenchmarkArgumentDictionary["directory"]
        databaseName = usedBenchmarkArgumentDictionary["args"].name
        labelsNames = [usedBenchmarkArgumentDictionary["LABELS_DICTIONARY"][0],
                       usedBenchmarkArgumentDictionary["LABELS_DICTIONARY"][1]]
        publishMetricsGraphs(metricsScores, directory, databaseName, labelsNames)
        publishExampleErrors(exampleErrors, directory, databaseName, labelsNames)
        biclassResults[iteridex][str(classifierPositive) + str(classifierNegative)]["metricsScores"] = metricsScores
        biclassResults[iteridex][str(classifierPositive) + str(classifierNegative)]["exampleErrors"] = exampleErrors
    logging.debug("Done:\t Analzing all biclass resuls")
    return biclassResults


def genMetricsScoresMulticlass(results, trueLabels, metrics, argumentsDictionaries):
    """Used to add all the metrics scores to the multiclass result structure  for each clf and each iteration"""

    logging.debug("Start:\t Getting multiclass scores for each metric")
    # TODO : Metric score for train and test
    for metric in metrics:
        metricModule = getattr(Metrics, metric[0])
        for iterIndex, iterResults in enumerate(results):
            for argumentsDictionary in argumentsDictionaries:
                if argumentsDictionary["flag"][0]==iterIndex:
                    classificationIndices = argumentsDictionary["classificationIndices"]
            trainIndices, testIndices, multiclassTestIndices = classificationIndices
            for classifierName, resultDictionary in iterResults.items():
                if not "metricsScores" in resultDictionary:
                    results[iterIndex][classifierName]["metricsScores"]={}
                trainScore = metricModule.score(trueLabels[trainIndices],resultDictionary["labels"][trainIndices], multiclass=True)
                testScore = metricModule.score(trueLabels[multiclassTestIndices],
                                               resultDictionary["labels"][multiclassTestIndices],
                                               multiclass=True)
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


def publishMulticlassScores(multiclassResults, metrics, statsIter, direcories, databaseName, minSize=10):
    for iterIndex in range(statsIter):
        directory = direcories[iterIndex]
        for metric in metrics:
            logging.debug("Start:\t Multiclass score graph generation for "+metric[0])
            classifiersNames = np.array([classifierName for classifierName in multiclassResults[iterIndex].keys()])
            trainScores = np.array([multiclassResults[iterIndex][classifierName]["metricsScores"][metric[0]][0]
                                for classifierName in classifiersNames])
            validationScores = np.array([multiclassResults[iterIndex][classifierName]["metricsScores"][metric[0]][1]
                                for classifierName in classifiersNames])


            nbResults = classifiersNames.shape[0]
            fileName = directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-" + metric[0] + ".png"

            plotMetricScores(trainScores, validationScores, classifiersNames, nbResults, metric[0], fileName, tag=" multiclass")

            logging.debug("Done:\t Multiclass score graph generation for " + metric[0])


def publishMulticlassExmapleErrors(multiclassResults, directories, databaseName, minSize=10):
    for iterIndex, multiclassResult in enumerate(multiclassResults):
        directory = directories[iterIndex]
        logging.debug("Start:\t Label analysis figure generation")
        nbClassifiers = len(multiclassResult)
        nbExamples = len(list(multiclassResult.values())[0]["errorOnExample"])
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
        cmap = mpl.colors.ListedColormap(['red', 'green'])
        bounds = [-0.5, 0.5, 1.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
        plt.title('Errors depending on the classifier')
        ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
        labels = classifiersNames
        plt.xticks(ticks, labels, rotation="vertical")
        red_patch = mpatches.Patch(color='red', label='Classifier failed')
        green_patch = mpatches.Patch(color='green', label='Classifier succeded')
        plt.legend(handles=[red_patch, green_patch], bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=2)
        fig.tight_layout()
        fig.savefig(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + databaseName +"-error_analysis.png", bbox_inches="tight")
        plt.close()
        logging.debug("Done:\t Label analysis figure generation")

        logging.debug("Start:\t Error by example figure generation")
        errorOnExamples = -1*np.sum(data, axis=1)/nbIter+nbClassifiers
        np.savetxt(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-clf_errors_doubled.csv", data, delimiter=",")
        np.savetxt(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-example_errors.csv", temp_data, delimiter=",")
        fig, ax = plt.subplots()
        x = np.arange(nbExamples)
        plt.bar(x, errorOnExamples)
        plt.ylim([0,nbClassifiers])
        plt.title("Number of classifiers that failed to classify each example")
        fig.savefig(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + databaseName +"-example_errors.png")
        plt.close()
        logging.debug("Done:\t Error by example figure generation")


def analyzeMulticlass(results, statsIter, benchmarkArgumentDictionaries, nbExamples, nbLabels, multiclassLabels,
                      metrics, classificationIndices, directories):
    """Used to transform one versus one results in multiclass results and to publish it"""
    multiclassResults = [{} for _ in range(statsIter)]

    for flag, result in results:
        iterIndex = flag[0]
        classifierPositive = flag[1][0]
        classifierNegative = flag[1][1]

        for benchmarkArgumentDictionary in benchmarkArgumentDictionaries:
            if benchmarkArgumentDictionary["flag"] == flag:
                trainIndices, testIndices, testMulticlassIndices = benchmarkArgumentDictionary["classificationIndices"]

        for classifierResult in result:
            classifierName = classifierResult.get_classifier_name()
            if classifierName not in multiclassResults[iterIndex]:
                multiclassResults[iterIndex][classifierName] = np.zeros((nbExamples, nbLabels),dtype=int)
            for exampleIndex in trainIndices:
                label = classifierResult.full_labels_pred[exampleIndex]
                if label == 1:
                    multiclassResults[iterIndex][classifierName][exampleIndex, classifierPositive] += 1
                else:
                    multiclassResults[iterIndex][classifierName][exampleIndex, classifierNegative] += 1
            for multiclassIndex, exampleIndex in enumerate(testMulticlassIndices):
                label = classifierResult.y_test_multiclass_pred[multiclassIndex]
                if label == 1:
                    multiclassResults[iterIndex][classifierName][exampleIndex, classifierPositive] += 1
                else:
                    multiclassResults[iterIndex][classifierName][exampleIndex, classifierNegative] += 1

    for iterIndex, multiclassiterResult in enumerate(multiclassResults):
        for key, value in multiclassiterResult.items():
            multiclassResults[iterIndex][key] = {"labels": np.argmax(value, axis=1)}

    multiclassResults = genMetricsScoresMulticlass(multiclassResults, multiclassLabels, metrics, benchmarkArgumentDictionaries)
    multiclassResults = getErrorOnLabelsMulticlass(multiclassResults, multiclassLabels)

    publishMulticlassScores(multiclassResults, metrics, statsIter, directories, benchmarkArgumentDictionaries[0]["args"].name)
    publishMulticlassExmapleErrors(multiclassResults, directories, benchmarkArgumentDictionaries[0]["args"].name)
    return multiclassResults

def numpy_mean_and_std(scores_array):
    return np.mean(scores_array, axis=1), np.std(scores_array, axis=1)


def publishIterBiclassMetricsScores(iterResults, directory, labelsDictionary, classifiersDict, dataBaseName, statsIter, minSize=10):
    for labelsCombination, iterResult in iterResults.items():
        currentDirectory = directory+ labelsDictionary[int(labelsCombination[0])]+"-vs-"+labelsDictionary[int(labelsCombination[1])]+"/"
        if not os.path.exists(os.path.dirname(currentDirectory+"a")):
            try:
                os.makedirs(os.path.dirname(currentDirectory+"a"))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        for metricName, scores in iterResult["metricsScores"].items():
            trainMeans, trainSTDs = numpy_mean_and_std(scores["trainScores"])
            testMeans, testSTDs = numpy_mean_and_std(scores["testScores"])

            names = np.array([name for name in classifiersDict.keys()])
            fileName = currentDirectory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + dataBaseName + "-Mean_on_" + str(statsIter) + "_iter-" + metricName + ".png"
            nbResults = names.shape[0]

            plotMetricScores(trainScores=trainMeans, testScores=testMeans, names=names, nbResults=nbResults,
                             metricName=metricName, fileName=fileName, tag=" averaged",
                             train_STDs=trainSTDs, test_STDs=testSTDs)



def iterCmap(statsIter):
    cmapList = ["red", "0.0"]
    for i in range(statsIter):
        cmapList.append(str(float((i+1))/statsIter))
    cmap = mpl.colors.ListedColormap(cmapList)
    bounds = [-100*statsIter-0.5, -0.5]
    for i in range(statsIter):
        bounds.append(i+0.5)
    bounds.append(statsIter+0.5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def publishIterBiclassExampleErrors(iterResults, directory, labelsDictionary, classifiersDict, statsIter, minSize=10):
    for labelsCombination, combiResults in iterResults.items():
        currentDirectory = directory+ labelsDictionary[int(labelsCombination[0])]+"-vs-"+labelsDictionary[int(labelsCombination[1])]+"/"
        reversedClassifiersDict = dict((value, key) for key, value in classifiersDict.items())
        classifiersNames = [reversedClassifiersDict[i] for i in range(len(classifiersDict))]

        logging.debug("Start:\t Global label analysis figure generation")
        nbExamples = combiResults["errorOnExamples"].shape[1]
        nbClassifiers = combiResults["errorOnExamples"].shape[0]
        figWidth = max(nbClassifiers / 2, minSize)
        figHeight = max(nbExamples / 20, minSize)
        figKW = {"figsize": (figWidth, figHeight)}
        fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
        data = np.transpose(combiResults["errorOnExamples"])
        cmap, norm = iterCmap(statsIter)
        cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
        plt.title('Errors depending on the classifier')
        ticks = np.arange(nbClassifiers)
        plt.xticks(ticks, classifiersNames, rotation="vertical")
        cbar = fig.colorbar(cax, ticks=[-100*statsIter/2, 0, statsIter])
        cbar.ax.set_yticklabels(['Unseen', 'Always Wrong', 'Always Right'])
        fig.tight_layout()
        fig.savefig(currentDirectory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-error_analysis.png")
        plt.close()
        logging.debug("Done:\t Global label analysis figure generation")

        logging.debug("Start:\t Global error by example figure generation")
        errorOnExamples = -1 * np.sum(data, axis=1) + (nbClassifiers*statsIter)
        np.savetxt(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors.csv", data, delimiter=",")
        np.savetxt(currentDirectory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", errorOnExamples, delimiter=",")
        fig, ax = plt.subplots()
        x = np.arange(nbExamples)
        plt.bar(x, errorOnExamples)
        plt.ylim([0,nbClassifiers*statsIter])
        plt.title("Number of classifiers that failed to classify each example")
        fig.savefig(currentDirectory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-example_errors.png")
        plt.close()
        logging.debug("Done:\t Global error by example figure generation")


def publishIterMulticlassMetricsScores(iterMulticlassResults, classifiersNames, dataBaseName, directory, statsIter, minSize=10):
    for metricName, scores in iterMulticlassResults["metricsScores"].items():

        trainMeans, trainSTDs = numpy_mean_and_std(scores["trainScores"])
        testMeans, testSTDs = numpy_mean_and_std(scores["testScores"])

        nbResults = classifiersNames.shape[0]

        fileName = directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-" + dataBaseName + "-Mean_on_" + str(statsIter) + "_iter-" + metricName + ".png"

        plotMetricScores(trainScores=trainMeans, testScores=testMeans, names=classifiersNames, nbResults=nbResults,
                         metricName=metricName, fileName=fileName, tag=" averaged multiclass",
                         train_STDs=trainSTDs, test_STDs=testSTDs)



def publishIterMulticlassExampleErrors(iterMulticlassResults, directory, classifiersNames, statsIter, minSize=10):

    logging.debug("Start:\t Global label analysis figure generation")
    nbExamples = iterMulticlassResults["errorOnExamples"].shape[1]
    nbClassifiers = iterMulticlassResults["errorOnExamples"].shape[0]
    figWidth = max(nbClassifiers / 2, minSize)
    figHeight = max(nbExamples / 20, minSize)
    figKW = {"figsize": (figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    data = np.transpose(iterMulticlassResults["errorOnExamples"])
    cax = plt.imshow(-data, interpolation='none', cmap="Greys", aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbClassifiers)
    plt.xticks(ticks, classifiersNames, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[0, -statsIter])
    cbar.ax.set_yticklabels(['Always Wrong', 'Always Right'])
    fig.tight_layout()
    fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
    plt.close()
    logging.debug("Done:\t Global label analysis figure generation")

    logging.debug("Start:\t Global error by example figure generation")
    errorOnExamples = -1 * np.sum(data, axis=1) + (nbClassifiers*statsIter)
    np.savetxt(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-clf_errors.csv", data, delimiter=",")
    np.savetxt(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-example_errors.csv", errorOnExamples, delimiter=",")
    fig, ax = plt.subplots()
    x = np.arange(nbExamples)
    plt.bar(x, errorOnExamples)
    plt.ylim([0,nbClassifiers*statsIter])
    plt.title("Number of classifiers that failed to classify each example")
    fig.savefig(directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-example_errors.png")
    plt.close()
    logging.debug("Done:\t Global error by example figure generation")


def analyzebiclassIter(biclassResults, metrics, statsIter, directory, labelsDictionary, dataBaseName, nbExamples):
    iterBiclassResults = {}
    classifiersDict = {}
    for iterIndex, biclassResult in enumerate(biclassResults):
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
                                                                                        nbExamples),
                                                                                       dtype=int)
                if metric[0] not in iterBiclassResults[labelsComination]["metricsScores"]:
                    iterBiclassResults[labelsComination]["metricsScores"][metric[0]]= {"trainScores":
                                                                                           np.zeros((nbClassifiers, statsIter)),
                                                                                       "testScores":
                                                                                           np.zeros((nbClassifiers, statsIter))}
                for classifierName, trainScore, testScore in zip(results["metricsScores"][metric[0]]["classifiersNames"],
                                                                 results["metricsScores"][metric[0]]["trainScores"],
                                                                 results["metricsScores"][metric[0]]["testScores"],
                                                                 ):
                    iterBiclassResults[labelsComination]["metricsScores"][metric[0]]["trainScores"][classifiersDict[classifierName], iterIndex] = trainScore
                    iterBiclassResults[labelsComination]["metricsScores"][metric[0]]["testScores"][classifiersDict[classifierName], iterIndex] = testScore
            for classifierName, errorOnExample in results["exampleErrors"].items():
                iterBiclassResults[labelsComination]["errorOnExamples"][classifiersDict[classifierName], :] += errorOnExample
    publishIterBiclassMetricsScores(iterBiclassResults, directory, labelsDictionary, classifiersDict, dataBaseName, statsIter)
    publishIterBiclassExampleErrors(iterBiclassResults, directory, labelsDictionary, classifiersDict, statsIter)


def analyzeIterMulticlass(multiclassResults, directory, statsIter, metrics, dataBaseName, nbExamples):
    """Used to mean the multiclass results on the iterations executed with different random states"""

    logging.debug("Start:\t Getting mean results for multiclass classification")
    iterMulticlassResults = {}
    nbClassifiers = len(multiclassResults[0])
    iterMulticlassResults["errorOnExamples"] = np.zeros((nbClassifiers,nbExamples),dtype=int)
    iterMulticlassResults["metricsScores"] = {}
    classifiersNames = []
    for iterIndex, multiclassResult in enumerate(multiclassResults):
        for classifierName, classifierResults in multiclassResult.items():
            if classifierName not in classifiersNames:
                classifiersNames.append(classifierName)
            classifierIndex = classifiersNames.index(classifierName)
            for metric in metrics:
                if metric[0] not in iterMulticlassResults["metricsScores"]:
                    iterMulticlassResults["metricsScores"][metric[0]] = {"trainScores":
                                                                             np.zeros((nbClassifiers, statsIter)),
                                                                         "testScores":
                                                                             np.zeros((nbClassifiers, statsIter))}
                iterMulticlassResults["metricsScores"][metric[0]]["trainScores"][classifierIndex, iterIndex] = classifierResults["metricsScores"][metric[0]][0]
                iterMulticlassResults["metricsScores"][metric[0]]["testScores"][classifierIndex, iterIndex] = classifierResults["metricsScores"][metric[0]][1]
            iterMulticlassResults["errorOnExamples"][classifierIndex, :] += classifierResults["errorOnExample"]
    logging.debug("Start:\t Getting mean results for multiclass classification")

    classifiersNames = np.array(classifiersNames)
    publishIterMulticlassMetricsScores(iterMulticlassResults, classifiersNames, dataBaseName, directory, statsIter)
    publishIterMulticlassExampleErrors(iterMulticlassResults, directory, classifiersNames, statsIter)


def getResults(results, statsIter, nbMulticlass, benchmarkArgumentDictionaries, multiclassLabels, metrics, classificationIndices, directories, directory, labelsDictionary, nbExamples, nbLabels):
    """Used to analyze the results of the previous benchmarks"""
    dataBaseName = benchmarkArgumentDictionaries[0]["args"].name
    if statsIter > 1:
        if nbMulticlass > 1:
            biclassResults = analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics)
            multiclassResults = analyzeMulticlass(results, statsIter, benchmarkArgumentDictionaries, nbExamples, nbLabels, multiclassLabels,
                                                  metrics, classificationIndices, directories)
            analyzebiclassIter(biclassResults, metrics, statsIter, directory, labelsDictionary, dataBaseName, nbExamples)
            analyzeIterMulticlass(multiclassResults, directory, statsIter, metrics, dataBaseName, nbExamples)
        else:
            biclassResults = analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics)
            analyzebiclassIter(biclassResults, metrics, statsIter, directory, labelsDictionary, dataBaseName, nbExamples)
    else:
        if nbMulticlass>1:
            biclassResults = analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics)
            multiclassResults = analyzeMulticlass(results, statsIter, benchmarkArgumentDictionaries, nbExamples, nbLabels, multiclassLabels,
                                                  metrics, classificationIndices, directories)
        else:
            biclassResults = analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics)





# def genFusionName(type_, a, b, c):
#     """Used to generate fusion classifiers names"""
#     if type_ == "Fusion" and a["fusionType"] != "EarlyFusion":
#         return "Late-" + str(a["fusionMethod"])
#     elif type_ == "Fusion" and a["fusionType"] != "LateFusion":
#         return "Early-" + a["fusionMethod"] + "-" + a["classifiersNames"]
#
#
# def genNamesFromRes(mono, multi):
#     """Used to generate classifiers names list (inthe right order) from mono- and multi-view preds"""
#     names = [res[1][0] + "-" + res[1][1][-1] for res in mono]
#     names += [type_ if type_ != "Fusion" else genFusionName(type_, a, b, c) for type_, a, b, c in multi]
#     return names
#
#
# def resultAnalysis(benchmark, results, name, times, metrics, directory, minSize=10):
#     """Used to generate bar graphs of all the classifiers scores for each metric """
#     mono, multi = results
#     for metric in metrics:
#         logging.debug("Start:\t Score graph generation for "+metric[0])
#         names = genNamesFromRes(mono, multi)
#         nbResults = len(mono) + len(multi)
#         validationScores = [float(res[1][2][metric[0]][1]) for res in mono]
#         validationScores += [float(scores[metric[0]][1]) for a, b, scores, c in multi]
#         trainScores = [float(res[1][2][metric[0]][0]) for res in mono]
#         trainScores += [float(scores[metric[0]][0]) for a, b, scores, c in multi]
#
#         validationScores = np.array(validationScores)
#         trainScores = np.array(trainScores)
#         names = np.array(names)
#         sorted_indices = np.argsort(validationScores)
#         validationScores = validationScores[sorted_indices]
#         trainScores = trainScores[sorted_indices]
#         names = names[sorted_indices]
#
#         size = nbResults
#         if nbResults < minSize:
#             size = minSize
#         figKW = {"figsize" : (size, 3.0/4*size+2.0)}
#         f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
#         barWidth= 0.35
#         ax.set_title(metric[0] + "\n on validation set for each classifier")
#         rects = ax.bar(range(nbResults), validationScores, barWidth, color="r", )
#         rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth, color="0.7", )
#         autolabel(rects, ax)
#         autolabel(rect2, ax)
#         ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
#         ax.set_ylim(-0.1, 1.1)
#         ax.set_xticks(np.arange(nbResults) + barWidth)
#         ax.set_xticklabels(names, rotation="vertical")
#         plt.tight_layout()
#         f.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + name + "-" + metric[0] + ".png")
#         plt.close()
#         logging.debug("Done:\t Score graph generation for " + metric[0])
#
#
# def analyzeLabels(labelsArrays, realLabels, results, directory, minSize = 10):
#     """Used to generate a graph showing errors on each example depending on classifier"""
#     logging.debug("Start:\t Label analysis figure generation")
#     mono, multi = results
#     classifiersNames = genNamesFromRes(mono, multi)
#     nbClassifiers = len(classifiersNames)
#     nbExamples = realLabels.shape[0]
#     nbIter = 2
#     data = np.zeros((nbExamples, nbClassifiers * nbIter))
#     tempData = np.array([labelsArray == realLabels for labelsArray in np.transpose(labelsArrays)]).astype(int)
#     for classifierIndex in range(nbClassifiers):
#         for iterIndex in range(nbIter):
#             data[:, classifierIndex * nbIter + iterIndex] = tempData[classifierIndex, :]
#     figWidth = max(nbClassifiers/2, minSize)
#     figHeight = max(nbExamples/20, minSize)
#     figKW = {"figsize":(figWidth, figHeight)}
#     fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
#     cmap = mpl.colors.ListedColormap(['red', 'green'])
#     bounds = [-0.5, 0.5, 1.5]
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#
#     cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm, aspect='auto')
#     plt.title('Errors depending on the classifier')
#     ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
#     labels = classifiersNames
#     plt.xticks(ticks, labels, rotation="vertical")
#     cbar = fig.colorbar(cax, ticks=[0, 1])
#     cbar.ax.set_yticklabels(['Wrong', ' Right'])
#     fig.tight_layout()
#     fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
#     plt.close()
#     logging.debug("Done:\t Label analysis figure generation")
#
#     logging.debug("Start:\t Error by example figure generation")
#     errorOnExamples = -1*np.sum(data, axis=1)/nbIter+nbClassifiers
#     np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors.csv", data, delimiter=",")
#     np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", errorOnExamples, delimiter=",")
#     fig, ax = plt.subplots()
#     x = np.arange(nbExamples)
#     plt.bar(x, errorOnExamples)
#     plt.ylim([0,nbClassifiers])
#     plt.title("Number of classifiers that failed to classify each example")
#     fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.png")
#     plt.close()
#     logging.debug("Done:\t Error by example figure generation")
#     return data
#
#
# def analyzeIterLabels(labelsAnalysisList, directory, classifiersNames, minSize=10):
#     """Used to generate a graph showing errors on each example depending on classifierusing a score
#      if multiple iterations"""
#     logging.debug("Start:\t Global label analysis figure generation")
#     nbExamples = labelsAnalysisList[0].shape[0]
#     nbClassifiers = len(classifiersNames)
#     nbIter = 2
#
#     figWidth = max(nbClassifiers / 2, minSize)
#     figHeight = max(nbExamples / 20, minSize)
#     figKW = {"figsize": (figWidth, figHeight)}
#     fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
#     data = sum(labelsAnalysisList)
#     cax = plt.imshow(-data, interpolation='none', cmap="Greys", aspect='auto')
#     plt.title('Errors depending on the classifier')
#     ticks = np.arange(nbIter/2-0.5, nbClassifiers * nbIter, nbIter)
#     plt.xticks(ticks, classifiersNames, rotation="vertical")
#     cbar = fig.colorbar(cax, ticks=[0, -len(labelsAnalysisList)])
#     cbar.ax.set_yticklabels(['Always Wrong', 'Always Right'])
#     fig.tight_layout()
#     fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-error_analysis.png")
#     plt.close()
#     logging.debug("Done:\t Global label analysis figure generation")
#     logging.debug("Start:\t Global error by example figure generation")
#     errorOnExamples = -1 * np.sum(data, axis=1) / nbIter + (nbClassifiers*len(labelsAnalysisList))
#     np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-clf_errors.csv", data, delimiter=",")
#     np.savetxt(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.csv", errorOnExamples, delimiter=",")
#     fig, ax = plt.subplots()
#     x = np.arange(nbExamples)
#     plt.bar(x, errorOnExamples)
#     plt.ylim([0,nbClassifiers*len(labelsAnalysisList)])
#     plt.title("Number of classifiers that failed to classify each example")
#     fig.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-example_errors.png")
#     plt.close()
#     logging.debug("Done:\t Global error by example figure generation")
#
#
# def genFig(iterResults, metric, nbResults, names, nbMono, minSize=10):
#     """Used to generate the bar graph representing the mean scores of each classifiers if multiple iteration
#      with different random states"""
#     nbIter = len(iterResults)
#     validationScores = np.zeros((nbIter, nbResults))
#     trainScores = np.zeros((nbIter, nbResults))
#     for iterIndex, iterResult in enumerate(iterResults):
#         mono, multi = iterResult
#         validationScores[iterIndex, :nbMono] = np.array([float(res[1][2][metric[0]][1]) for res in mono])
#         validationScores[iterIndex, nbMono:] = np.array([float(scores[metric[0]][1]) for a, b, scores, c in multi])
#         trainScores[iterIndex, :nbMono] = np.array([float(res[1][2][metric[0]][0]) for res in mono])
#         trainScores[iterIndex, nbMono:] = np.array([float(scores[metric[0]][0]) for a, b, scores, c in multi])
#
#     validationSTDs = np.std(validationScores, axis=0)
#     trainSTDs = np.std(trainScores, axis=0)
#     validationMeans = np.mean(validationScores, axis=0)
#     trainMeans = np.mean(trainScores, axis=0)
#     size=nbResults
#     if nbResults<minSize:
#         size=minSize
#     figKW = {"figsize" : (size, 3.0/4*size+2.0)}
#     f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
#     barWidth = 0.35  # the width of the bars
#     sorted_indices = np.argsort(validationMeans)
#     validationMeans = validationMeans[sorted_indices]
#     validationSTDs = validationSTDs[sorted_indices]
#     trainSTDs = trainSTDs[sorted_indices]
#     trainMeans = trainMeans[sorted_indices]
#     names = np.array(names)[sorted_indices]
#
#     ax.set_title(metric[0] + " for each classifier")
#     rects = ax.bar(range(nbResults), validationMeans, barWidth, color="r", yerr=validationSTDs)
#     rect2 = ax.bar(np.arange(nbResults) + barWidth, trainMeans, barWidth, color="0.7", yerr=trainSTDs)
#     autolabel(rects, ax)
#     autolabel(rect2, ax)
#     ax.set_ylim(-0.1, 1.1)
#     ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
#     ax.set_xticks(np.arange(nbResults) + barWidth)
#     ax.set_xticklabels(names, rotation="vertical")
#     f.tight_layout()
#
#     return f
#
#
# def analyzeIterResults(iterResults, name, metrics, directory):
#     nbResults = len(iterResults[0][0]) + len(iterResults[0][1])
#     nbMono = len(iterResults[0][0])
#     nbIter = len(iterResults)
#     names = genNamesFromRes(iterResults[0][0], iterResults[0][1])
#     for metric in metrics:
#         logging.debug("Start:\t Global score graph generation for " + metric[0])
#         figure = genFig(iterResults, metric, nbResults, names, nbMono)
#         figure.savefig(directory + time.strftime("%Y%m%d-%H%M%S") + "-" + name + "-Mean_on_"
#                        + str(nbIter) + "_iter-" + metric[0] + ".png")
#         logging.debug("Done:\t Global score graph generation for " + metric[0])
