# Import built-in modules
import errno
import logging
import os
import time

import matplotlib as mpl
from matplotlib.patches import Patch
# Import third party modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import own Modules
from . import metrics

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def plot_results_noise(directory, noise_results, metric_to_plot, name, width=0.1):
    avail_colors = ["tab:blue", "tab:orange", "tab:brown", "tab:gray",
                    "tab:olive", "tab:red", ]
    colors ={}
    lengend_patches = []
    noise_levels = np.array([noise_level for noise_level, _ in noise_results])
    df = pd.DataFrame(columns=['noise_level', 'classifier_name', 'mean_score', 'score_std'], )
    if len(noise_results)>1:
        width = np.min(np.diff(noise_levels))
    for noise_level, noise_result in noise_results:
        classifiers_names, meaned_metrics, metric_stds =  [], [], []
        for noise_result in noise_result:
            classifier_name = noise_result[0].split("-")[0]
            if noise_result[1] is metric_to_plot:
                classifiers_names.append(classifier_name)
                meaned_metrics.append(noise_result[2])
                metric_stds.append(noise_result[3])
                if classifier_name not in colors:
                    try:
                        colors[classifier_name] = avail_colors.pop(0)
                    except IndexError:
                        colors[classifier_name] = "k"
        classifiers_names, meaned_metrics, metric_stds = np.array(classifiers_names), np.array(meaned_metrics), np.array(metric_stds)
        sorted_indices = np.argsort(-meaned_metrics)
        for index in sorted_indices:
            row = pd.DataFrame(
                {'noise_level':noise_level, 'classifier_name':classifiers_names[index], 'mean_score':meaned_metrics[index],
                         'score_std':metric_stds[index]}, index=[0])
            df = pd.concat([df, row])
            plt.bar(noise_level, meaned_metrics[index], yerr=metric_stds[index], width=0.5*width, label=classifiers_names[index], color=colors[classifiers_names[index]])
    for classifier_name, color in colors.items():
        lengend_patches.append(Patch(facecolor=color, label=classifier_name))
    plt.legend(handles=lengend_patches, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.ylabel(metric_to_plot)
    plt.title(name)
    plt.xticks(noise_levels)
    plt.xlabel("Noise level")
    plt.savefig(directory+name+"_noise_analysis.png")
    plt.close()
    df.to_csv(directory+name+"_noise_analysis.csv")



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
                    "%.2f" % height + u'\u00B1' + "%.2f" % std[rectIndex],
                    weight=weight,
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
        errorOnExamples = np.equal(classifierResult.full_labels_pred,
                                   groud_truth).astype(int)
        unseenExamples = np.where(groud_truth == -100)[0]
        errorOnExamples[unseenExamples] = -100
        exampleErrors[classifierResult.get_classifier_name()] = {
            "errorOnExamples": errorOnExamples}

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
    size = nb_results * multiplier
    if size < min_size:
        size = min_size
    fig_kwargs = {"figsize": (size, size / 3)}
    return fig_kwargs, bar_width


def sort_by_test_score(train_scores, test_scores, names, train_STDs=None,
                       test_STDs=None):
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


def     plotMetricScores(trainScores, testScores, names, nbResults, metricName,
                     fileName,
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
    tag : str
        Some text to personalize the title, must start with a whitespace.
    train_STDs : np.array of floats or None
        The array containing the standard deviations for the averaged scores on the training set.
    test_STDs : np.array of floats or None
        The array containing the standard deviations for the averaged scores on the testing set.

    Returns
    -------
    """

    figKW, barWidth = get_fig_size(nbResults)

    names, trainScores, testScores, train_STDs, test_STDs = sort_by_test_score(
        trainScores, testScores, names,
        train_STDs, test_STDs)

    f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    ax.set_title(metricName + "\n" + tag + " scores for each classifier")

    rects = ax.bar(range(nbResults), testScores, barWidth, color="0.1",
                   yerr=test_STDs)
    rect2 = ax.bar(np.arange(nbResults) + barWidth, trainScores, barWidth,
                   color="0.8", yerr=train_STDs)
    autolabel(rects, ax, set=1, std=test_STDs)
    autolabel(rect2, ax, set=2, std=train_STDs)

    ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(nbResults) + barWidth)
    ax.set_xticklabels(names, rotation="vertical")

    try:
        plt.tight_layout()
    except:
        pass
    f.savefig(fileName + '.png', transparent=True)
    plt.close()
    import pandas as pd
    if train_STDs is None:
        dataframe = pd.DataFrame(np.transpose(np.concatenate((
            trainScores.reshape((trainScores.shape[0], 1)),
            testScores.reshape((trainScores.shape[0], 1))), axis=1)),
            columns=names)
    else:
        dataframe = pd.DataFrame(np.transpose(np.concatenate((
            trainScores.reshape((trainScores.shape[0], 1)),
            train_STDs.reshape((trainScores.shape[0], 1)),
            testScores.reshape((trainScores.shape[0], 1)),
            test_STDs.reshape((trainScores.shape[0], 1))), axis=1)),
            columns=names)
    dataframe.to_csv(fileName + ".csv")


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
    results=[]
    for metricName, metricScores in metricsScores.items():
        logging.debug(
            "Start:\t Biclass score graph generation for " + metricName)

        nbResults = len(metricScores["testScores"])

        fileName = directory + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-" + "_vs_".join(
            labelsNames) + "-" + metricName

        plotMetricScores(np.array(metricScores["trainScores"]),
                         np.array(metricScores["testScores"]),
                         np.array(metricScores["classifiersNames"]), nbResults,
                         metricName, fileName,
                         tag=" " + " vs ".join(labelsNames))

        logging.debug(
            "Done:\t Biclass score graph generation for " + metricName)
        results+=[[classifiersName, metricName, testMean, testSTD] for classifiersName, testMean, testSTD in zip(np.array(metricScores["classifiersNames"]), np.array(metricScores["testScores"]), np.zeros(len(np.array(metricScores["testScores"]))))]
    return results

def iterCmap(statsIter):
    r"""Used to generate a colormap that will have a tick for each iteration : the whiter the better.

    Parameters
    ----------
    statsIter : int
        The number of statistical iterations.

    Returns
    -------
    cmap : matplotlib.colors.ListedColorMap object
        The colormap.
    norm : matplotlib.colors.BoundaryNorm object
        The bounds for the colormap.
    """
    cmapList = ["red", "0.0"] + [str(float((i + 1)) / statsIter) for i in
                                 range(statsIter)]
    cmap = mpl.colors.ListedColormap(cmapList)
    bounds = [-100 * statsIter - 0.5, -0.5]
    for i in range(statsIter):
        bounds.append(i + 0.5)
    bounds.append(statsIter + 0.5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def publish2Dplot(data, classifiersNames, nbClassifiers, nbExamples, nbCopies,
                  fileName, minSize=10,
                  width_denominator=2.0, height_denominator=20.0, statsIter=1):
    r"""Used to generate a 2D plot of the errors.

    Parameters
    ----------
    data : np.array of shape `(nbClassifiers, nbExamples)`
        A matrix with zeros where the classifier failed to classifiy the example, ones where it classified it well
        and -100 if the example was not classified.
    classifiersNames : list of str
        The names of the classifiers.
    nbClassifiers : int
        The number of classifiers.
    nbExamples : int
        The number of examples.
    nbCopies : int
        The number of times the data is copied (classifier wise) in order for the figure to be more readable
    fileName : str
        The name of the file in which the figure will be saved ("error_analysis_2D.png" will be added at the end)
    minSize : int, optinal, default: 10
        The minimum width and height of the figure.
    width_denominator : float, optional, default: 1.0
        To obtain the image width, the number of classifiers will be divided by this number.
    height_denominator : float, optional, default: 1.0
        To obtain the image width, the number of examples will be divided by this number.
    statsIter : int, optional, default: 1
        The number of statistical iterations realized.

    Returns
    -------
    """
    figWidth = max(nbClassifiers / width_denominator, minSize)
    figHeight = max(nbExamples / height_denominator, minSize)
    figKW = {"figsize": (figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    cmap, norm = iterCmap(statsIter)
    cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm,
                     aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbCopies / 2 - 0.5, nbClassifiers * nbCopies, nbCopies)
    labels = classifiersNames
    plt.xticks(ticks, labels, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[-100 * statsIter / 2, 0, statsIter])
    cbar.ax.set_yticklabels(['Unseen', 'Always Wrong', 'Always Right'])
    fig.tight_layout()
    fig.savefig(fileName + "error_analysis_2D.png", bbox_inches="tight", transparent=True)
    plt.close()


def publishErrorsBarPlot(errorOnExamples, nbClassifiers, nbExamples, fileName):
    r"""Used to generate a barplot of the muber of classifiers that failed to classify each examples

    Parameters
    ----------
    errorOnExamples : np.array of shape `(nbExamples,)`
        An array counting how many classifiers failed to classifiy each examples.
    classifiersNames : list of str
        The names of the classifiers.
    nbClassifiers : int
        The number of classifiers.
    nbExamples : int
        The number of examples.
    fileName : str
        The name of the file in which the figure will be saved ("error_analysis_2D.png" will be added at the end)

    Returns
    -------
    """
    fig, ax = plt.subplots()
    x = np.arange(nbExamples)
    plt.bar(x, errorOnExamples)
    plt.ylim([0, nbClassifiers])
    plt.title("Number of classifiers that failed to classify each example")
    fig.savefig(fileName + "error_analysis_bar.png", transparent=True)
    plt.close()


def gen_error_data(example_errors, base_file_name, nbCopies=2):
    r"""Used to format the error data in order to plot it efficiently. The data is saves in a `.csv` file.

    Parameters
    ----------
    example_errors : dict of dicts of np.arrays
        A dictionary conatining all the useful data. Organized as :
        `example_errors[<classifier_name>]["errorOnExamples"]` is a np.array of ints with a
        - 1 if the classifier `<classifier_name>` classifier well the example,
        - 0 if it fail to classify the example,
        - -100 if it did not classify the example (multiclass one versus one).
    base_file_name : list of str
        The name of the file in which the figure will be saved ("2D_plot_data.csv" and "bar_plot_data.csv" will
        be added at the end).
    nbCopies : int, optinal, default: 2
        The number of times the data is copied (classifier wise) in order for the figure to be more readable.


    Returns
    -------
    nbClassifiers : int
        Number of different classifiers.
    nbExamples : int
        NUmber of examples.
    nbCopies : int
        The number of times the data is copied (classifier wise) in order for the figure to be more readable.
    classifiersNames : list of strs
        The names fo the classifiers.
    data : np.array of shape `(nbClassifiers, nbExamples)`
        A matrix with zeros where the classifier failed to classifiy the example, ones where it classified it well
        and -100 if the example was not classified.
    errorOnExamples : np.array of shape `(nbExamples,)`
        An array counting how many classifiers failed to classifiy each examples.
    """
    nbClassifiers = len(example_errors)
    nbExamples = len(list(example_errors.values())[0]["errorOnExamples"])
    classifiersNames = example_errors.keys()

    data = np.zeros((nbExamples, nbClassifiers * nbCopies))
    temp_data = np.zeros((nbExamples, nbClassifiers))
    for classifierIndex, (classifierName, errorOnExamples) in enumerate(
            example_errors.items()):
        for iterIndex in range(nbCopies):
            data[:, classifierIndex * nbCopies + iterIndex] = errorOnExamples[
                "errorOnExamples"]
            temp_data[:, classifierIndex] = errorOnExamples["errorOnExamples"]
    errorOnExamples = -1 * np.sum(data, axis=1) / nbCopies + nbClassifiers

    np.savetxt(base_file_name + "2D_plot_data.csv", data, delimiter=",")
    np.savetxt(base_file_name + "bar_plot_data.csv", temp_data, delimiter=",")

    return nbClassifiers, nbExamples, nbCopies, classifiersNames, data, errorOnExamples


def publishExampleErrors(exampleErrors, directory, databaseName, labelsNames):
    logging.debug("Start:\t Biclass Label analysis figure generation")

    base_file_name = directory + time.strftime(
        "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-" + "_vs_".join(
        labelsNames) + "-"

    nbClassifiers, nbExamples, nCopies, classifiersNames, data, errorOnExamples = gen_error_data(
        exampleErrors,
        base_file_name)

    publish2Dplot(data, classifiersNames, nbClassifiers, nbExamples, nCopies,
                  base_file_name)

    publishErrorsBarPlot(errorOnExamples, nbClassifiers, nbExamples,
                         base_file_name)

    logging.debug("Done:\t Biclass Label analysis figures generation")


def get_arguments(benchmarkArgumentDictionaries, flag):
    r"""Used to get the arguments passed to the benchmark executing function corresponding to the flag of a
    biclass experimentation.

    Parameters
    ----------
    flag : list
        The needed experimentation's flag.
    benchmarkArgumentDictionaries : list of dicts
        The list of all the arguments passed to the benchmark executing functions.

    Returns
    -------
    benchmarkArgumentDictionary : dict
        All the arguments passed to the benchmark executing function for the needed experimentation.
    """
    for benchmarkArgumentDictionary in benchmarkArgumentDictionaries:
        if benchmarkArgumentDictionary["flag"] == flag:
            return benchmarkArgumentDictionary


def analyzeBiclass(results, benchmarkArgumentDictionaries, statsIter, metrics):
    r"""Used to extract and format the results of the different biclass experimentations performed.

    Parameters
    ----------
    results : list
        The result list returned by the bencmark execution function. For each executed benchmark, contains
        a flag & a result element.
        The flag is a way to identify to which benchmark the results belong, formatted this way :
        `flag = iterIndex, [classifierPositive, classifierNegative]` with
        - `iterIndex` the index of the statistical iteration
        - `[classifierPositive, classifierNegative]` the indices of the labels considered positive and negative
        by the classifier (mainly useful for one versus one multiclass classification).
    benchmarkArgumentDictionaries : list of dicts
        The list of all the arguments passed to the benchmark executing functions.
    statsIter : int
        The number of statistical iterations.
    metrics : list of lists
        THe list containing the metrics and their configuration.

    Returns
    -------
    biclassResults : list of dicts of dicts
        The list contains a dictionary for each statistical iteration. This dictionary contains a dictionary for each
        label combination, regrouping the scores for each metrics and the information useful to plot errors on examples.
    """
    logging.debug("Srart:\t Analzing all biclass resuls")
    biclassResults = [{} for _ in range(statsIter)]

    for flag, result in results:
        iteridex, [classifierPositive, classifierNegative] = flag

        arguments = get_arguments(benchmarkArgumentDictionaries, flag)

        metricsScores = getMetricsScoresBiclass(metrics, result)
        exampleErrors = getExampleErrorsBiclass(arguments["labels"], result)

        directory = arguments["directory"]

        databaseName = arguments["args"].name
        labelsNames = [arguments["LABELS_DICTIONARY"][0],
                       arguments["LABELS_DICTIONARY"][1]]

        results = publishMetricsGraphs(metricsScores, directory, databaseName,
                             labelsNames)
        publishExampleErrors(exampleErrors, directory, databaseName,
                             labelsNames)

        biclassResults[iteridex][
            str(classifierPositive) + str(classifierNegative)] = {
            "metricsScores": metricsScores,
            "exampleErrors": exampleErrors}

    logging.debug("Done:\t Analzing all biclass resuls")
    return results, biclassResults


def genMetricsScoresMulticlass(results, trueLabels, metrics,
                               argumentsDictionaries):
    """Used to add all the metrics scores to the multiclass result structure  for each clf and each iteration"""

    logging.debug("Start:\t Getting multiclass scores for each metric")

    for metric in metrics:
        metricModule = getattr(metrics, metric[0])
        for iterIndex, iterResults in enumerate(results):

            for argumentsDictionary in argumentsDictionaries:
                if argumentsDictionary["flag"][0] == iterIndex:
                    classificationIndices = argumentsDictionary[
                        "classificationIndices"]
            trainIndices, testIndices, multiclassTestIndices = classificationIndices

            for classifierName, resultDictionary in iterResults.items():
                if not "metricsScores" in resultDictionary:
                    results[iterIndex][classifierName]["metricsScores"] = {}
                trainScore = metricModule.score(trueLabels[trainIndices],
                                                resultDictionary["labels"][
                                                    trainIndices],
                                                multiclass=True)
                testScore = metricModule.score(
                    trueLabels[multiclassTestIndices],
                    resultDictionary["labels"][multiclassTestIndices],
                    multiclass=True)
                results[iterIndex][classifierName]["metricsScores"][
                    metric[0]] = [trainScore, testScore]
    logging.debug("Done:\t Getting multiclass scores for each metric")
    return results


def getErrorOnLabelsMulticlass(multiclassResults, multiclassLabels):
    """Used to add all the arrays showing on which example there is an error for each clf and each iteration"""

    logging.debug("Start:\t Getting errors on each example for each classifier")

    for iterIndex, iterResults in enumerate(multiclassResults):
        for classifierName, classifierResults in iterResults.items():
            errorOnExamples = classifierResults["labels"] == multiclassLabels
            multiclassResults[iterIndex][classifierName][
                "errorOnExamples"] = errorOnExamples.astype(int)

    logging.debug("Done:\t Getting errors on each example for each classifier")

    return multiclassResults


def publishMulticlassScores(multiclassResults, metrics, statsIter, direcories,
                            databaseName):
    results=[]
    for iterIndex in range(statsIter):
        directory = direcories[iterIndex]
        for metric in metrics:
            logging.debug(
                "Start:\t Multiclass score graph generation for " + metric[0])
            classifiersNames = np.array([classifierName for classifierName in
                                         multiclassResults[iterIndex].keys()])
            trainScores = np.array([multiclassResults[iterIndex][
                                        classifierName]["metricsScores"][
                                        metric[0]][0]
                                    for classifierName in classifiersNames])
            validationScores = np.array([multiclassResults[iterIndex][
                                             classifierName]["metricsScores"][
                                             metric[0]][1]
                                         for classifierName in
                                         classifiersNames])

            nbResults = classifiersNames.shape[0]
            fileName = directory + time.strftime(
                "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-" + metric[
                           0] + ".png"

            plotMetricScores(trainScores, validationScores, classifiersNames,
                             nbResults, metric[0], fileName, tag=" multiclass")

            logging.debug(
                "Done:\t Multiclass score graph generation for " + metric[0])
            results+=[[classifiersName, metric, testMean, testSTD] for classifiersName, testMean, testSTD in zip(classifiersNames, validationScores, np.zeros(len(validationScores)))]
    return results


def publishMulticlassExmapleErrors(multiclassResults, directories,
                                   databaseName):
    for iterIndex, multiclassResult in enumerate(multiclassResults):
        directory = directories[iterIndex]
        logging.debug("Start:\t Multiclass Label analysis figure generation")

        base_file_name = directory + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-"

        nbClassifiers, nbExamples, nCopies, classifiersNames, data, errorOnExamples = gen_error_data(
            multiclassResult,
            base_file_name)

        publish2Dplot(data, classifiersNames, nbClassifiers, nbExamples,
                      nCopies, base_file_name)

        publishErrorsBarPlot(errorOnExamples, nbClassifiers, nbExamples,
                             base_file_name)

        logging.debug("Done:\t Multiclass Label analysis figure generation")


def analyzeMulticlass(results, statsIter, benchmarkArgumentDictionaries,
                      nbExamples, nbLabels, multiclassLabels,
                      metrics, classificationIndices, directories):
    """Used to transform one versus one results in multiclass results and to publish it"""
    multiclassResults = [{} for _ in range(statsIter)]

    for flag, result in results:
        iterIndex = flag[0]
        classifierPositive = flag[1][0]
        classifierNegative = flag[1][1]

        for benchmarkArgumentDictionary in benchmarkArgumentDictionaries:
            if benchmarkArgumentDictionary["flag"] == flag:
                trainIndices, testIndices, testMulticlassIndices = \
                benchmarkArgumentDictionary["classificationIndices"]

        for classifierResult in result:
            classifierName = classifierResult.get_classifier_name()
            if classifierName not in multiclassResults[iterIndex]:
                multiclassResults[iterIndex][classifierName] = np.zeros(
                    (nbExamples, nbLabels), dtype=int)
            for exampleIndex in trainIndices:
                label = classifierResult.full_labels_pred[exampleIndex]
                if label == 1:
                    multiclassResults[iterIndex][classifierName][
                        exampleIndex, classifierPositive] += 1
                else:
                    multiclassResults[iterIndex][classifierName][
                        exampleIndex, classifierNegative] += 1
            for multiclassIndex, exampleIndex in enumerate(
                    testMulticlassIndices):
                label = classifierResult.y_test_multiclass_pred[multiclassIndex]
                if label == 1:
                    multiclassResults[iterIndex][classifierName][
                        exampleIndex, classifierPositive] += 1
                else:
                    multiclassResults[iterIndex][classifierName][
                        exampleIndex, classifierNegative] += 1

    for iterIndex, multiclassiterResult in enumerate(multiclassResults):
        for key, value in multiclassiterResult.items():
            multiclassResults[iterIndex][key] = {
                "labels": np.argmax(value, axis=1)}

    multiclassResults = genMetricsScoresMulticlass(multiclassResults,
                                                   multiclassLabels, metrics,
                                                   benchmarkArgumentDictionaries)
    multiclassResults = getErrorOnLabelsMulticlass(multiclassResults,
                                                   multiclassLabels)

    results = publishMulticlassScores(multiclassResults, metrics, statsIter, directories,
                            benchmarkArgumentDictionaries[0]["args"].name)
    publishMulticlassExmapleErrors(multiclassResults, directories,
                                   benchmarkArgumentDictionaries[0][
                                       "args"].name)
    return results, multiclassResults


def numpy_mean_and_std(scores_array):
    return np.mean(scores_array, axis=1), np.std(scores_array, axis=1)


def publishIterBiclassMetricsScores(iterResults, directory, labelsDictionary,
                                    classifiersDict, dataBaseName, statsIter,
                                    minSize=10):
    results=[]
    for labelsCombination, iterResult in iterResults.items():
        currentDirectory = directory + labelsDictionary[
            int(labelsCombination[0])] + "-vs-" + labelsDictionary[
                               int(labelsCombination[1])] + "/"
        if not os.path.exists(os.path.dirname(currentDirectory + "a")):
            try:
                os.makedirs(os.path.dirname(currentDirectory + "a"))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        for metricName, scores in iterResult["metricsScores"].items():
            trainMeans, trainSTDs = numpy_mean_and_std(scores["trainScores"])
            testMeans, testSTDs = numpy_mean_and_std(scores["testScores"])

            names = np.array([name for name in classifiersDict.keys()])
            fileName = currentDirectory + time.strftime(
                "%Y_%m_%d-%H_%M_%S") + "-" + dataBaseName + "-Mean_on_" + str(
                statsIter) + "_iter-" + metricName + ".png"
            nbResults = names.shape[0]

            plotMetricScores(trainScores=trainMeans, testScores=testMeans,
                             names=names, nbResults=nbResults,
                             metricName=metricName, fileName=fileName,
                             tag=" averaged",
                             train_STDs=trainSTDs, test_STDs=testSTDs)
            results+=[[classifiersName, metricName, testMean, testSTD] for classifiersName, testMean, testSTD in zip(names, testMeans, testSTDs)]
    return results


def gen_error_dat_glob(combiResults, statsIter, base_file_name):
    nbExamples = combiResults["errorOnExamples"].shape[1]
    nbClassifiers = combiResults["errorOnExamples"].shape[0]
    data = np.transpose(combiResults["errorOnExamples"])
    errorOnExamples = -1 * np.sum(data, axis=1) + (nbClassifiers * statsIter)
    np.savetxt(base_file_name + "clf_errors.csv", data, delimiter=",")
    np.savetxt(base_file_name + "example_errors.csv", errorOnExamples,
               delimiter=",")
    return nbExamples, nbClassifiers, data, errorOnExamples


def publishIterBiclassExampleErrors(iterResults, directory, labelsDictionary,
                                    classifiersDict, statsIter, minSize=10):
    for labelsCombination, combiResults in iterResults.items():
        base_file_name = directory + labelsDictionary[
            int(labelsCombination[0])] + "-vs-" + \
                         labelsDictionary[
                             int(labelsCombination[1])] + "/" + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-"
        classifiersNames = [classifierName for classifierName in
                            classifiersDict.values()]
        logging.debug(
            "Start:\t Global biclass label analysis figure generation")

        nbExamples, nbClassifiers, data, errorOnExamples = gen_error_dat_glob(
            combiResults, statsIter, base_file_name)

        publish2Dplot(data, classifiersNames, nbClassifiers, nbExamples, 1,
                      base_file_name, statsIter=statsIter)

        publishErrorsBarPlot(errorOnExamples, nbClassifiers * statsIter,
                             nbExamples, base_file_name)

        logging.debug(
            "Done:\t Global biclass label analysis figures generation")


def publishIterMulticlassMetricsScores(iterMulticlassResults, classifiersNames,
                                       dataBaseName, directory, statsIter,
                                       minSize=10):
    results = []
    for metricName, scores in iterMulticlassResults["metricsScores"].items():
        trainMeans, trainSTDs = numpy_mean_and_std(scores["trainScores"])
        testMeans, testSTDs = numpy_mean_and_std(scores["testScores"])

        nbResults = classifiersNames.shape[0]

        fileName = directory + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-" + dataBaseName + "-Mean_on_" + str(
            statsIter) + "_iter-" + metricName + ".png"

        plotMetricScores(trainScores=trainMeans, testScores=testMeans,
                         names=classifiersNames, nbResults=nbResults,
                         metricName=metricName, fileName=fileName,
                         tag=" averaged multiclass",
                         train_STDs=trainSTDs, test_STDs=testSTDs)

        results+=[[classifiersName, metricName,testMean, testSTD] for classifiersName, testMean, testSTD in zip(classifiersNames, testMeans, testSTDs)]
    return results


def publishIterMulticlassExampleErrors(iterMulticlassResults, directory,
                                       classifiersNames, statsIter, minSize=10):
    logging.debug(
        "Start:\t Global multiclass label analysis figures generation")
    base_file_name = directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-"

    nbExamples, nbClassifiers, data, errorOnExamples = gen_error_dat_glob(
        iterMulticlassResults, statsIter, base_file_name)

    publish2Dplot(data, classifiersNames, nbClassifiers, nbExamples, 1,
                  base_file_name, statsIter=statsIter)

    publishErrorsBarPlot(errorOnExamples, nbClassifiers * statsIter, nbExamples,
                         base_file_name)

    logging.debug("Done:\t Global multiclass label analysis figures generation")


def gen_classifiers_dict(results, metrics):
    classifiersDict = dict((classifierName, classifierIndex)
                           for classifierIndex, classifierName
                           in enumerate(
        results[0][list(results[0].keys())[0]]["metricsScores"][metrics[0][0]][
            "classifiersNames"]))
    return classifiersDict, len(classifiersDict)


def add_new_labels_combination(iterBiclassResults, labelsComination,
                               nbClassifiers, nbExamples):
    if labelsComination not in iterBiclassResults:
        iterBiclassResults[labelsComination] = {}
        iterBiclassResults[labelsComination]["metricsScores"] = {}

        iterBiclassResults[labelsComination]["errorOnExamples"] = np.zeros(
            (nbClassifiers,
             nbExamples),
            dtype=int)
    return iterBiclassResults


def add_new_metric(iterBiclassResults, metric, labelsComination, nbClassifiers,
                   statsIter):
    if metric[0] not in iterBiclassResults[labelsComination]["metricsScores"]:
        iterBiclassResults[labelsComination]["metricsScores"][metric[0]] = {
            "trainScores":
                np.zeros((nbClassifiers, statsIter)),
            "testScores":
                np.zeros((nbClassifiers, statsIter))}
    return iterBiclassResults


def analyzebiclassIter(biclassResults, metrics, statsIter, directory,
                       labelsDictionary, dataBaseName, nbExamples):
    """Used to format the results in order to plot the mean results on the iterations"""
    iterBiclassResults = {}
    classifiersDict, nbClassifiers = gen_classifiers_dict(biclassResults,
                                                          metrics)

    for iterIndex, biclassResult in enumerate(biclassResults):
        for labelsComination, results in biclassResult.items():
            for metric in metrics:

                iterBiclassResults = add_new_labels_combination(
                    iterBiclassResults, labelsComination, nbClassifiers,
                    nbExamples)
                iterBiclassResults = add_new_metric(iterBiclassResults, metric,
                                                    labelsComination,
                                                    nbClassifiers, statsIter)

                metric_results = results["metricsScores"][metric[0]]
                for classifierName, trainScore, testScore in zip(
                        metric_results["classifiersNames"],
                        metric_results["trainScores"],
                        metric_results["testScores"], ):
                    iterBiclassResults[labelsComination]["metricsScores"][
                        metric[0]]["trainScores"][
                        classifiersDict[classifierName], iterIndex] = trainScore
                    iterBiclassResults[labelsComination]["metricsScores"][
                        metric[0]]["testScores"][
                        classifiersDict[classifierName], iterIndex] = testScore

            for classifierName, errorOnExample in results[
                "exampleErrors"].items():
                iterBiclassResults[labelsComination]["errorOnExamples"][
                classifiersDict[classifierName], :] += errorOnExample[
                    "errorOnExamples"]

    results = publishIterBiclassMetricsScores(iterBiclassResults, directory,
                                    labelsDictionary, classifiersDict,
                                    dataBaseName, statsIter)
    publishIterBiclassExampleErrors(iterBiclassResults, directory,
                                    labelsDictionary, classifiersDict,
                                    statsIter)
    return results

def analyzeIterMulticlass(multiclassResults, directory, statsIter, metrics,
                          dataBaseName, nbExamples):
    """Used to mean the multiclass results on the iterations executed with different random states"""

    logging.debug("Start:\t Getting mean results for multiclass classification")
    iterMulticlassResults = {}
    nbClassifiers = len(multiclassResults[0])
    iterMulticlassResults["errorOnExamples"] = np.zeros(
        (nbClassifiers, nbExamples), dtype=int)
    iterMulticlassResults["metricsScores"] = {}
    classifiersNames = []
    for iterIndex, multiclassResult in enumerate(multiclassResults):
        for classifierName, classifierResults in multiclassResult.items():
            if classifierName not in classifiersNames:
                classifiersNames.append(classifierName)
            classifierIndex = classifiersNames.index(classifierName)
            for metric in metrics:
                if metric[0] not in iterMulticlassResults["metricsScores"]:
                    iterMulticlassResults["metricsScores"][metric[0]] = {
                        "trainScores":
                            np.zeros((nbClassifiers, statsIter)),
                        "testScores":
                            np.zeros((nbClassifiers, statsIter))}
                iterMulticlassResults["metricsScores"][metric[0]][
                    "trainScores"][classifierIndex, iterIndex] = \
                classifierResults["metricsScores"][metric[0]][0]
                iterMulticlassResults["metricsScores"][metric[0]]["testScores"][
                    classifierIndex, iterIndex] = \
                classifierResults["metricsScores"][metric[0]][1]
            iterMulticlassResults["errorOnExamples"][classifierIndex, :] += \
            classifierResults["errorOnExamples"]
    logging.debug("Start:\t Getting mean results for multiclass classification")

    classifiersNames = np.array(classifiersNames)
    results = publishIterMulticlassMetricsScores(iterMulticlassResults, classifiersNames,
                                       dataBaseName, directory, statsIter)
    publishIterMulticlassExampleErrors(iterMulticlassResults, directory,
                                       classifiersNames, statsIter)
    return results


def getResults(results, statsIter, nbMulticlass, benchmarkArgumentDictionaries,
               multiclassLabels, metrics,
               classificationIndices, directories, directory, labelsDictionary,
               nbExamples, nbLabels):
    """Used to analyze the results of the previous benchmarks"""
    dataBaseName = benchmarkArgumentDictionaries[0]["args"].name
    results_means_std, biclassResults = analyzeBiclass(results, benchmarkArgumentDictionaries,
                                    statsIter, metrics)

    if nbMulticlass > 1:
        results_means_std, multiclassResults = analyzeMulticlass(results, statsIter,
                                              benchmarkArgumentDictionaries,
                                              nbExamples, nbLabels,
                                              multiclassLabels, metrics,
                                              classificationIndices,
                                              directories)
    if statsIter > 1:
        results_means_std = analyzebiclassIter(biclassResults, metrics, statsIter, directory,
                           labelsDictionary, dataBaseName, nbExamples)
        if nbMulticlass > 1:
            results_means_std = analyzeIterMulticlass(multiclassResults, directory, statsIter,
                                  metrics, dataBaseName, nbExamples)
    return results_means_std
