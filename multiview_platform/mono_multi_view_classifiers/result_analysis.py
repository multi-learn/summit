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


def get_metrics_scores_biclass(metrics, results):
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
        -`metricScores[metric_name]["classifiers_names"]` is a list of all the classifiers available for this metric,
        -`metricScores[metric_name]["train_scores"]` is a list of all the available classifiers scores on the train set,
        -`metricScores[metric_name]["test_scores"]` is a list of all the available classifiers scores on the test set.
    """
    metrics_scores = {}

    for metric in metrics:
        classifiers_names = []
        train_scores = []
        test_scores = []

        for classifierResult in results:
            train_scores.append(classifierResult.metrics_scores[metric[0]][0])
            test_scores.append(classifierResult.metrics_scores[metric[0]][1])
            classifiers_names.append(classifierResult.get_classifier_name())

        metrics_scores[metric[0]] = {"classifiers_names": classifiers_names,
                                    "train_scores": train_scores,
                                    "test_scores": test_scores}
    return metrics_scores


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
    example_errors : dict of np.array
        For each classifier, has an entry with a `np.array` over the examples, with a 1 if the examples was
        well-classified, a 0 if not and if it's multiclass classification, a -100 if the examples was not seen during
        the one versus one classification.
    """
    example_errors = {}

    for classifierResult in results:
        error_on_examples = np.equal(classifierResult.full_labels_pred,
                                   groud_truth).astype(int)
        unseenExamples = np.where(groud_truth == -100)[0]
        error_on_examples[unseenExamples] = -100
        example_errors[classifierResult.get_classifier_name()] = {
            "error_on_examples": error_on_examples}

    return example_errors


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


def plotMetricScores(train_scores, test_scores, names, nb_results, metric_name,
                     file_name,
                     tag="", train_STDs=None, test_STDs=None):
    r"""Used to plot and save the score barplot for a specific metric.

    Parameters
    ----------
    train_scores : list or np.array of floats
        The scores of each classifier on the training set.
    test_scores : list or np.array of floats
        The scores of each classifier on the testing set.
    names : list or np.array of strs
        The names of all the classifiers.
    nb_results: int
        The number of classifiers to plot.
    metric_name : str
        The plotted metric's name
    file_name : str
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

    figKW, barWidth = get_fig_size(nb_results)

    names, train_scores, test_scores, train_STDs, test_STDs = sort_by_test_score(
        train_scores, test_scores, names,
        train_STDs, test_STDs)

    f, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    ax.set_title(metric_name + "\n" + tag + " scores for each classifier")

    rects = ax.bar(range(nb_results), test_scores, barWidth, color="0.1",
                   yerr=test_STDs)
    rect2 = ax.bar(np.arange(nb_results) + barWidth, train_scores, barWidth,
                   color="0.8", yerr=train_STDs)
    autolabel(rects, ax, set=1, std=test_STDs)
    autolabel(rect2, ax, set=2, std=train_STDs)
    print("nb_results", nb_results)
    ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(nb_results) + barWidth)
    ax.set_xticklabels(names, rotation="vertical")

    try:
        plt.tight_layout()
    except:
        pass
    f.savefig(file_name + '.png', transparent=True)
    plt.close()
    import pandas as pd
    if train_STDs is None:
        dataframe = pd.DataFrame(np.transpose(np.concatenate((
            train_scores.reshape((train_scores.shape[0], 1)),
            test_scores.reshape((train_scores.shape[0], 1))), axis=1)),
            columns=names)
    else:
        dataframe = pd.DataFrame(np.transpose(np.concatenate((
            train_scores.reshape((train_scores.shape[0], 1)),
            train_STDs.reshape((train_scores.shape[0], 1)),
            test_scores.reshape((train_scores.shape[0], 1)),
            test_STDs.reshape((train_scores.shape[0], 1))), axis=1)),
            columns=names)
    dataframe.to_csv(file_name + ".csv")


def publishMetricsGraphs(metrics_scores, directory, database_name, labels_names):
    r"""Used to sort the results (names and both scores) in descending test score order.

    Parameters
    ----------
    metrics_scores : dict of dicts of lists or np.arrays
        Keys : The names of the metrics.
        Values : The scores and names of each classifier .
    directory : str
        The path to the directory where the figures will be saved.
    database_name : str
        The name of the database on which the experiments where conducted.
    labels_names : list of strs
        The name corresponding to each numerical label.

    Returns
    -------
    results
    """
    results=[]
    for metric_name, metric_scores in metrics_scores.items():
        logging.debug(
            "Start:\t Biclass score graph generation for " + metric_name)

        nb_results = len(metric_scores["test_scores"])
        file_name = directory + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-" + database_name + "-" + "_vs_".join(
            labels_names) + "-" + metric_name

        plotMetricScores(np.array(metric_scores["train_scores"]),
                         np.array(metric_scores["test_scores"]),
                         np.array(metric_scores["classifiers_names"]), nb_results,
                         metric_name, file_name,
                         tag=" " + " vs ".join(labels_names))

        logging.debug(
            "Done:\t Biclass score graph generation for " + metric_name)
        results+=[[classifiers_name, metric_name, testMean, testSTD]
                  for classifiers_name, testMean, testSTD in zip(np.array(metric_scores["classifiers_names"]),
                                                                 np.array(metric_scores["test_scores"]),
                                                                 np.zeros(len(np.array(metric_scores["test_scores"]))))]
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


def publish2Dplot(data, classifiers_names, nbClassifiers, nbExamples, nbCopies,
                  fileName, minSize=10,
                  width_denominator=2.0, height_denominator=20.0, stats_iter=1):
    r"""Used to generate a 2D plot of the errors.

    Parameters
    ----------
    data : np.array of shape `(nbClassifiers, nbExamples)`
        A matrix with zeros where the classifier failed to classifiy the example, ones where it classified it well
        and -100 if the example was not classified.
    classifiers_names : list of str
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
    stats_iter : int, optional, default: 1
        The number of statistical iterations realized.

    Returns
    -------
    """
    figWidth = max(nbClassifiers / width_denominator, minSize)
    figHeight = max(nbExamples / height_denominator, minSize)
    figKW = {"figsize": (figWidth, figHeight)}
    fig, ax = plt.subplots(nrows=1, ncols=1, **figKW)
    cmap, norm = iterCmap(stats_iter)
    cax = plt.imshow(data, interpolation='none', cmap=cmap, norm=norm,
                     aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(nbCopies / 2 - 0.5, nbClassifiers * nbCopies, nbCopies)
    labels = classifiers_names
    plt.xticks(ticks, labels, rotation="vertical")
    cbar = fig.colorbar(cax, ticks=[-100 * stats_iter / 2, 0, stats_iter])
    cbar.ax.set_yticklabels(['Unseen', 'Always Wrong', 'Always Right'])
    fig.tight_layout()
    fig.savefig(fileName + "error_analysis_2D.png", bbox_inches="tight", transparent=True)
    plt.close()


def publishErrorsBarPlot(error_on_examples, nbClassifiers, nbExamples, fileName):
    r"""Used to generate a barplot of the muber of classifiers that failed to classify each examples

    Parameters
    ----------
    error_on_examples : np.array of shape `(nbExamples,)`
        An array counting how many classifiers failed to classifiy each examples.
    classifiers_names : list of str
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
    plt.bar(x, error_on_examples)
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
        `example_errors[<classifier_name>]["error_on_examples"]` is a np.array of ints with a
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
    classifiers_names : list of strs
        The names fo the classifiers.
    data : np.array of shape `(nbClassifiers, nbExamples)`
        A matrix with zeros where the classifier failed to classifiy the example, ones where it classified it well
        and -100 if the example was not classified.
    error_on_examples : np.array of shape `(nbExamples,)`
        An array counting how many classifiers failed to classifiy each examples.
    """
    nbClassifiers = len(example_errors)
    nbExamples = len(list(example_errors.values())[0]["error_on_examples"])
    classifiers_names = example_errors.keys()

    data = np.zeros((nbExamples, nbClassifiers * nbCopies))
    temp_data = np.zeros((nbExamples, nbClassifiers))
    for classifierIndex, (classifier_name, error_on_examples) in enumerate(
            example_errors.items()):
        for iter_index in range(nbCopies):
            data[:, classifierIndex * nbCopies + iter_index] = error_on_examples[
                "error_on_examples"]
            temp_data[:, classifierIndex] = error_on_examples["error_on_examples"]
    error_on_examples = -1 * np.sum(data, axis=1) / nbCopies + nbClassifiers

    np.savetxt(base_file_name + "2D_plot_data.csv", data, delimiter=",")
    np.savetxt(base_file_name + "bar_plot_data.csv", temp_data, delimiter=",")

    return nbClassifiers, nbExamples, nbCopies, classifiers_names, data, error_on_examples


def publishExampleErrors(example_errors, directory, databaseName, labels_names):
    logging.debug("Start:\t Biclass Label analysis figure generation")

    base_file_name = directory + time.strftime(
        "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-" + "_vs_".join(
        labels_names) + "-"

    nbClassifiers, nbExamples, nCopies, classifiers_names, data, error_on_examples = gen_error_data(
        example_errors,
        base_file_name)

    publish2Dplot(data, classifiers_names, nbClassifiers, nbExamples, nCopies,
                  base_file_name)

    publishErrorsBarPlot(error_on_examples, nbClassifiers, nbExamples,
                         base_file_name)

    logging.debug("Done:\t Biclass Label analysis figures generation")


def get_arguments(benchmark_argument_dictionaries, flag):
    r"""Used to get the arguments passed to the benchmark executing function corresponding to the flag of a
    biclass experimentation.

    Parameters
    ----------
    flag : list
        The needed experimentation's flag.
    benchmark_argument_dictionaries : list of dicts
        The list of all the arguments passed to the benchmark executing functions.

    Returns
    -------
    benchmarkArgumentDictionary : dict
        All the arguments passed to the benchmark executing function for the needed experimentation.
    """
    for benchmarkArgumentDictionary in benchmark_argument_dictionaries:
        if benchmarkArgumentDictionary["flag"] == flag:
            return benchmarkArgumentDictionary


def analyze_biclass(results, benchmark_argument_dictionaries, stats_iter, metrics):
    r"""Used to extract and format the results of the different biclass experimentations performed.

    Parameters
    ----------
    results : list
        The result list returned by the bencmark execution function. For each executed benchmark, contains
        a flag & a result element.
        The flag is a way to identify to which benchmark the results belong, formatted this way :
        `flag = iter_index, [classifierPositive, classifierNegative]` with
        - `iter_index` the index of the statistical iteration
        - `[classifierPositive, classifierNegative]` the indices of the labels considered positive and negative
        by the classifier (mainly useful for one versus one multiclass classification).
    benchmark_argument_dictionaries : list of dicts
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
    biclass_results = [{} for _ in range(stats_iter)]

    for flag, result in results:
        iteridex, [classifierPositive, classifierNegative] = flag

        arguments = get_arguments(benchmark_argument_dictionaries, flag)

        metrics_scores = get_metrics_scores_biclass(metrics, result)
        example_errors = getExampleErrorsBiclass(arguments["labels"], result)

        directory = arguments["directory"]

        database_name = arguments["args"]["Base"]["name"]
        labels_names = [arguments["labels_dictionary"][0],
                       arguments["labels_dictionary"][1]]

        results = publishMetricsGraphs(metrics_scores, directory, database_name,
                             labels_names)
        publishExampleErrors(example_errors, directory, database_name,
                             labels_names)

        biclass_results[iteridex][
            str(classifierPositive) + str(classifierNegative)] = {
            "metrics_scores": metrics_scores,
            "example_errors": example_errors}

    logging.debug("Done:\t Analzing all biclass resuls")
    return results, biclass_results


def gen_metrics_scores_multiclass(results, true_labels, metrics,
                                  arguments_dictionaries):
    """Used to add all the metrics scores to the multiclass result structure  for each clf and each iteration"""

    logging.debug("Start:\t Getting multiclass scores for each metric")

    for metric in metrics:
        metric_module = getattr(metrics, metric[0])
        for iter_index, iter_results in enumerate(results):

            for argumentsDictionary in arguments_dictionaries:
                if argumentsDictionary["flag"][0] == iter_index:
                    classification_indices = argumentsDictionary[
                        "classification_indices"]
            train_indices, test_indices, multiclass_test_indices = classification_indices

            for classifier_name, resultDictionary in iter_results.items():
                if not "metrics_scores" in resultDictionary:
                    results[iter_index][classifier_name]["metrics_scores"] = {}
                train_score = metric_module.score(true_labels[train_indices],
                                                resultDictionary["labels"][
                                                    train_indices],
                                                multiclass=True)
                test_score = metric_module.score(
                    true_labels[multiclass_test_indices],
                    resultDictionary["labels"][multiclass_test_indices],
                    multiclass=True)
                results[iter_index][classifier_name]["metrics_scores"][
                    metric[0]] = [train_score, test_score]
    logging.debug("Done:\t Getting multiclass scores for each metric")
    return results


def get_error_on_labels_multiclass(multiclass_results, multiclass_labels):
    """Used to add all the arrays showing on which example there is an error for each clf and each iteration"""

    logging.debug("Start:\t Getting errors on each example for each classifier")

    for iter_index, iter_results in enumerate(multiclass_results):
        for classifier_name, classifier_results in iter_results.items():
            error_on_examples = classifier_results["labels"] == multiclass_labels
            multiclass_results[iter_index][classifier_name][
                "error_on_examples"] = error_on_examples.astype(int)

    logging.debug("Done:\t Getting errors on each example for each classifier")

    return multiclass_results


def publishMulticlassScores(multiclass_results, metrics, stats_iter, direcories,
                            databaseName):
    results=[]
    for iter_index in range(stats_iter):
        directory = direcories[iter_index]
        for metric in metrics:
            logging.debug(
                "Start:\t Multiclass score graph generation for " + metric[0])
            classifiers_names = np.array([classifier_name for classifier_name in
                                         multiclass_results[iter_index].keys()])
            train_scores = np.array([multiclass_results[iter_index][
                                        classifier_name]["metrics_scores"][
                                        metric[0]][0]
                                    for classifier_name in classifiers_names])
            validationScores = np.array([multiclass_results[iter_index][
                                             classifier_name]["metrics_scores"][
                                             metric[0]][1]
                                         for classifier_name in
                                         classifiers_names])

            nbResults = classifiers_names.shape[0]
            fileName = directory + time.strftime(
                "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-" + metric[
                           0] + ".png"

            plotMetricScores(train_scores, validationScores, classifiers_names,
                             nbResults, metric[0], fileName, tag=" multiclass")

            logging.debug(
                "Done:\t Multiclass score graph generation for " + metric[0])
            results+=[[classifiersName, metric, testMean, testSTD] for classifiersName, testMean, testSTD in zip(classifiers_names, validationScores, np.zeros(len(validationScores)))]
    return results


def publishMulticlassExmapleErrors(multiclass_results, directories,
                                   databaseName):
    for iter_index, multiclassResult in enumerate(multiclass_results):
        directory = directories[iter_index]
        logging.debug("Start:\t Multiclass Label analysis figure generation")

        base_file_name = directory + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-" + databaseName + "-"

        nbClassifiers, nbExamples, nCopies, classifiers_names, data, error_on_examples = gen_error_data(
            multiclassResult,
            base_file_name)

        publish2Dplot(data, classifiers_names, nbClassifiers, nbExamples,
                      nCopies, base_file_name)

        publishErrorsBarPlot(error_on_examples, nbClassifiers, nbExamples,
                             base_file_name)

        logging.debug("Done:\t Multiclass Label analysis figure generation")


def analyzeMulticlass(results, stats_iter, benchmark_argument_dictionaries,
                      nb_examples, nb_labels, multiclass_labels,
                      metrics, classification_indices, directories):
    """Used to transform one versus one results in multiclass results and to publish it"""
    multiclass_results = [{} for _ in range(stats_iter)]

    for flag, result in results:
        iter_index = flag[0]
        classifierPositive = flag[1][0]
        classifierNegative = flag[1][1]

        for benchmarkArgumentDictionary in benchmark_argument_dictionaries:
            if benchmarkArgumentDictionary["flag"] == flag:
                trainIndices, testIndices, testMulticlassIndices = \
                benchmarkArgumentDictionary["classification_indices"]

        for classifierResult in result:
            classifier_name = classifierResult.get_classifier_name()
            if classifier_name not in multiclass_results[iter_index]:
                multiclass_results[iter_index][classifier_name] = np.zeros(
                    (nb_examples, nb_labels), dtype=int)
            for exampleIndex in trainIndices:
                label = classifierResult.full_labels_pred[exampleIndex]
                if label == 1:
                    multiclass_results[iter_index][classifier_name][
                        exampleIndex, classifierPositive] += 1
                else:
                    multiclass_results[iter_index][classifier_name][
                        exampleIndex, classifierNegative] += 1
            for multiclassIndex, exampleIndex in enumerate(
                    testMulticlassIndices):
                label = classifierResult.y_test_multiclass_pred[multiclassIndex]
                if label == 1:
                    multiclass_results[iter_index][classifier_name][
                        exampleIndex, classifierPositive] += 1
                else:
                    multiclass_results[iter_index][classifier_name][
                        exampleIndex, classifierNegative] += 1

    for iter_index, multiclassiterResult in enumerate(multiclass_results):
        for key, value in multiclassiterResult.items():
            multiclass_results[iter_index][key] = {
                "labels": np.argmax(value, axis=1)}

    multiclass_results = gen_metrics_scores_multiclass(multiclass_results,
                                                   multiclass_labels, metrics,
                                                   benchmark_argument_dictionaries)
    multiclass_results = get_error_on_labels_multiclass(multiclass_results,
                                                   multiclass_labels)

    results = publishMulticlassScores(multiclass_results, metrics, stats_iter, directories,
                            benchmark_argument_dictionaries[0]["args"]["Base"]["name"])
    publishMulticlassExmapleErrors(multiclass_results, directories,
                                   benchmark_argument_dictionaries[0][
                                       "args"].name)
    return results, multiclass_results


def numpy_mean_and_std(scores_array):
    return np.mean(scores_array, axis=1), np.std(scores_array, axis=1)


def publish_iter_biclass_metrics_scores(iter_results, directory, labels_dictionary,
                                    classifiers_dict, data_base_name, stats_iter,
                                    min_size=10):
    results=[]
    for labelsCombination, iterResult in iter_results.items():
        currentDirectory = directory + labels_dictionary[
            int(labelsCombination[0])] + "-vs-" + labels_dictionary[
                               int(labelsCombination[1])] + "/"
        if not os.path.exists(os.path.dirname(currentDirectory + "a")):
            try:
                os.makedirs(os.path.dirname(currentDirectory + "a"))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        for metricName, scores in iterResult["metrics_scores"].items():
            trainMeans, trainSTDs = numpy_mean_and_std(scores["train_scores"])
            testMeans, testSTDs = numpy_mean_and_std(scores["test_scores"])

            names = np.array([name for name in classifiers_dict.keys()])
            fileName = currentDirectory + time.strftime(
                "%Y_%m_%d-%H_%M_%S") + "-" + data_base_name + "-Mean_on_" + str(
                stats_iter) + "_iter-" + metricName + ".png"
            nbResults = names.shape[0]

            plotMetricScores(trainMeans, testMeans, names, nbResults,
                             metricName, fileName, tag=" averaged",
                             train_STDs=trainSTDs, test_STDs=testSTDs)
            results+=[[classifiersName, metricName, testMean, testSTD] for classifiersName, testMean, testSTD in zip(names, testMeans, testSTDs)]
    return results


def gen_error_dat_glob(combi_results, stats_iter, base_file_name):
    nbExamples = combi_results["error_on_examples"].shape[1]
    nbClassifiers = combi_results["error_on_examples"].shape[0]
    data = np.transpose(combi_results["error_on_examples"])
    error_on_examples = -1 * np.sum(data, axis=1) + (nbClassifiers * stats_iter)
    np.savetxt(base_file_name + "clf_errors.csv", data, delimiter=",")
    np.savetxt(base_file_name + "example_errors.csv", error_on_examples,
               delimiter=",")
    return nbExamples, nbClassifiers, data, error_on_examples


def publish_iter_biclass_example_errors(iter_results, directory, labels_dictionary,
                                        classifiers_dict, stats_iter, min_size=10):
    for labelsCombination, combiResults in iter_results.items():
        base_file_name = directory + labels_dictionary[
            int(labelsCombination[0])] + "-vs-" + \
                         labels_dictionary[
                             int(labelsCombination[1])] + "/" + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-"
        classifiers_names = [classifier_name for classifier_name in
                            classifiers_dict.values()]
        logging.debug(
            "Start:\t Global biclass label analysis figure generation")

        nbExamples, nbClassifiers, data, error_on_examples = gen_error_dat_glob(
            combiResults, stats_iter, base_file_name)

        publish2Dplot(data, classifiers_names, nbClassifiers, nbExamples, 1,
                      base_file_name, stats_iter=stats_iter)

        publishErrorsBarPlot(error_on_examples, nbClassifiers * stats_iter,
                             nbExamples, base_file_name)

        logging.debug(
            "Done:\t Global biclass label analysis figures generation")


def publish_iter_multiclass_metrics_scores(iter_multiclass_results, classifiers_names,
                                           data_base_name, directory, stats_iter,
                                           min_size=10):
    results = []
    for metric_name, scores in iter_multiclass_results["metrics_scores"].items():
        trainMeans, trainSTDs = numpy_mean_and_std(scores["train_scores"])
        testMeans, testSTDs = numpy_mean_and_std(scores["test_scores"])

        nb_results = classifiers_names.shape[0]

        file_name = directory + time.strftime(
            "%Y_%m_%d-%H_%M_%S") + "-" + data_base_name + "-Mean_on_" + str(
            stats_iter) + "_iter-" + metric_name + ".png"

        plotMetricScores(trainMeans, testMeans, classifiers_names, nb_results,
                         metric_name, file_name, tag=" averaged multiclass",
                         train_STDs=trainSTDs, test_STDs=testSTDs)

        results+=[[classifiers_name, metric_name,testMean, testSTD] for classifiers_name, testMean, testSTD in zip(classifiers_names, testMeans, testSTDs)]
    return results


def publish_iter_multiclass_example_errors(iter_multiclass_results, directory,
                                           classifiers_names, stats_iter, min_size=10):
    logging.debug(
        "Start:\t Global multiclass label analysis figures generation")
    base_file_name = directory + time.strftime("%Y_%m_%d-%H_%M_%S") + "-"

    nb_examples, nb_classifiers, data, error_on_examples = gen_error_dat_glob(
        iter_multiclass_results, stats_iter, base_file_name)

    publish2Dplot(data, classifiers_names, nb_classifiers, nb_examples, 1,
                  base_file_name, stats_iter=stats_iter)

    publishErrorsBarPlot(error_on_examples, nb_classifiers * stats_iter, nb_examples,
                         base_file_name)

    logging.debug("Done:\t Global multiclass label analysis figures generation")


def gen_classifiers_dict(results, metrics):
    classifiers_dict = dict((classifier_name, classifierIndex)
                           for classifierIndex, classifier_name
                           in enumerate(
        results[0][list(results[0].keys())[0]]["metrics_scores"][metrics[0][0]][
            "classifiers_names"]))
    return classifiers_dict, len(classifiers_dict)


def add_new_labels_combination(iterBiclassResults, labelsComination,
                               nbClassifiers, nbExamples):
    if labelsComination not in iterBiclassResults:
        iterBiclassResults[labelsComination] = {}
        iterBiclassResults[labelsComination]["metrics_scores"] = {}

        iterBiclassResults[labelsComination]["error_on_examples"] = np.zeros(
            (nbClassifiers,
             nbExamples),
            dtype=int)
    return iterBiclassResults


def add_new_metric(iter_biclass_results, metric, labels_combination, nb_classifiers,
                   stats_iter):
    if metric[0] not in iter_biclass_results[labels_combination]["metrics_scores"]:
        iter_biclass_results[labels_combination]["metrics_scores"][metric[0]] = {
            "train_scores":
                np.zeros((nb_classifiers, stats_iter)),
            "test_scores":
                np.zeros((nb_classifiers, stats_iter))}
    return iter_biclass_results


def analyzebiclass_iter(biclass_results, metrics, stats_iter, directory,
                       labels_dictionary, data_base_name, nb_examples):
    """Used to format the results in order to plot the mean results on the iterations"""
    iter_biclass_results = {}
    classifiers_dict, nb_classifiers = gen_classifiers_dict(biclass_results,
                                                          metrics)

    for iter_index, biclass_result in enumerate(biclass_results):
        for labelsComination, results in biclass_result.items():
            for metric in metrics:

                iter_biclass_results = add_new_labels_combination(
                    iter_biclass_results, labelsComination, nb_classifiers,
                    nb_examples)
                iter_biclass_results = add_new_metric(iter_biclass_results, metric,
                                                    labelsComination,
                                                    nb_classifiers, stats_iter)

                metric_results = results["metrics_scores"][metric[0]]
                for classifier_name, trainScore, testScore in zip(
                        metric_results["classifiers_names"],
                        metric_results["train_scores"],
                        metric_results["test_scores"], ):
                    iter_biclass_results[labelsComination]["metrics_scores"][
                        metric[0]]["train_scores"][
                        classifiers_dict[classifier_name], iter_index] = trainScore
                    iter_biclass_results[labelsComination]["metrics_scores"][
                        metric[0]]["test_scores"][
                        classifiers_dict[classifier_name], iter_index] = testScore
            for classifier_name, error_on_example in results[
                "example_errors"].items():
                iter_biclass_results[labelsComination]["error_on_examples"][
                classifiers_dict[classifier_name], :] += error_on_example[
                    "error_on_examples"]

    results = publish_iter_biclass_metrics_scores(
        iter_biclass_results, directory,
        labels_dictionary, classifiers_dict,
        data_base_name, stats_iter)
    publish_iter_biclass_example_errors(iter_biclass_results, directory,
                                        labels_dictionary, classifiers_dict,
                                        stats_iter)
    return results

def analyze_iter_multiclass(multiclass_results, directory, stats_iter, metrics,
                           data_base_name, nb_examples):
    """Used to mean the multiclass results on the iterations executed with different random states"""

    logging.debug("Start:\t Getting mean results for multiclass classification")
    iter_multiclass_results = {}
    nb_classifiers = len(multiclass_results[0])
    iter_multiclass_results["error_on_examples"] = np.zeros(
        (nb_classifiers, nb_examples), dtype=int)
    iter_multiclass_results["metrics_scores"] = {}
    classifiers_names = []
    for iter_index, multiclass_result in enumerate(multiclass_results):
        for classifier_name, classifier_results in multiclass_result.items():
            if classifier_name not in classifiers_names:
                classifiers_names.append(classifier_name)
            classifier_index = classifiers_names.index(classifier_name)
            for metric in metrics:
                if metric[0] not in iter_multiclass_results["metrics_scores"]:
                    iter_multiclass_results["metrics_scores"][metric[0]] = {
                        "train_scores":
                            np.zeros((nb_classifiers, stats_iter)),
                        "test_scores":
                            np.zeros((nb_classifiers, stats_iter))}
                iter_multiclass_results["metrics_scores"][metric[0]][
                    "train_scores"][classifier_index, iter_index] = \
                classifier_results["metrics_scores"][metric[0]][0]
                iter_multiclass_results["metrics_scores"][metric[0]]["test_scores"][
                    classifier_index, iter_index] = \
                classifier_results["metrics_scores"][metric[0]][1]
            iter_multiclass_results["error_on_examples"][classifier_index, :] += \
            classifier_results["error_on_examples"]
    logging.debug("Start:\t Getting mean results for multiclass classification")

    classifiers_names = np.array(classifiers_names)
    results = publish_iter_multiclass_metrics_scores(
        iter_multiclass_results, classifiers_names,
        data_base_name, directory, stats_iter)
    publish_iter_multiclass_example_errors(iter_multiclass_results, directory,
                                       classifiers_names, stats_iter)
    return results


def get_results(results, stats_iter, nb_multiclass, benchmark_argument_dictionaries,
               multiclass_labels, metrics,
               classification_indices, directories, directory, labels_dictionary,
               nb_examples, nb_labels):

    """Used to analyze the results of the previous benchmarks"""
    data_base_name = benchmark_argument_dictionaries[0]["args"]["Base"]["name"]
    results_means_std, biclass_results = analyze_biclass(results, benchmark_argument_dictionaries,
                                         stats_iter, metrics)

    if nb_multiclass > 1:
        results_means_std, multiclass_results = analyzeMulticlass(results, stats_iter,
                                              benchmark_argument_dictionaries,
                                              nb_examples, nb_labels,
                                              multiclass_labels, metrics,
                                              classification_indices,
                                              directories)
    if stats_iter > 1:
        results_means_std = analyzebiclass_iter(
            biclass_results, metrics, stats_iter, directory,
            labels_dictionary, data_base_name, nb_examples)
        if nb_multiclass > 1:
            results_means_std = analyze_iter_multiclass(multiclass_results, directory, stats_iter,
                                  metrics, data_base_name, nb_examples)
    return results_means_std
