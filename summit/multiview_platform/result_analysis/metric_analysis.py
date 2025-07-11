# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly

from ..utils.organization import secure_file_path


def get_metrics_scores(metrics, results, label_names):
    r"""Used to extract metrics scores in case of classification

    Parameters
    ----------
    metrics : dict
        The metrics names with configuration metrics[i][0] = name of metric i
    results : list of MonoviewResult and MultiviewResults objects
        A list containing all the results for all the monoview experimentations.

    Returns
    -------
    metricsScores : dict of dict of list
        Regroups all the scores for each metrics for each classifier and for
        the train and test sets.
        organized as :
        -`metricScores[metric_name]["classifiers_names"]` is a list of all the
        classifiers available for this metric,
        -`metricScores[metric_name]["train_scores"]` is a list of all the
        available classifiers scores on the train set,
        -`metricScores[metric_name]["test_scores"]` is a list of all the
        available classifiers scores on the test set.
    """
    classifier_names = []
    classifier_names = [classifier_result.get_classifier_name()
                        for classifier_result in results
                        if classifier_result.get_classifier_name()
                        not in classifier_names]
    metrics_scores = dict((metric, pd.DataFrame(data=np.zeros((2,
                                                               len(
                                                                   classifier_names))),
                                                index=["train", "test"],
                                                columns=classifier_names))
                          for metric in metrics.keys())
    class_metric_scores = dict((metric, pd.DataFrame(
        index=pd.MultiIndex.from_product([["train", "test"], label_names]),
        columns=classifier_names, dtype=float))
        for metric in metrics)

    for metric in metrics.keys():
        for classifier_result in results:
            metrics_scores[metric].loc[
                "train", classifier_result.get_classifier_name()] = \
                classifier_result.metrics_scores[metric][0]
            metrics_scores[metric].loc[
                "test", classifier_result.get_classifier_name()] = \
                classifier_result.metrics_scores[metric][1]
            for label_index, label_name in enumerate(label_names):
                class_metric_scores[metric].loc[(
                    "train",
                    label_name), classifier_result.get_classifier_name()] = \
                    classifier_result.class_metric_scores[metric][0][
                        label_index]
                class_metric_scores[metric].loc[(
                    "test",
                    label_name), classifier_result.get_classifier_name()] = \
                    classifier_result.class_metric_scores[metric][1][
                        label_index]

    return metrics_scores, class_metric_scores


def publish_metrics_graphs(metrics_scores, directory, database_name,
                           labels_names,
                           class_metric_scores):  # pragma: no cover
    r"""Used to sort the results (names and both scores) in descending test
    score order.

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
    results = []
    for metric_name in metrics_scores.keys():
        logging.info(
            "Start:\t Score graph generation for " + metric_name)
        train_scores, test_scores, classifier_names, \
            file_name, nb_results, results, \
            class_test_scores = init_plot(results, metric_name,
                                          metrics_scores[metric_name],
                                          directory,
                                          database_name,
                                          class_metric_scores[metric_name])

        plot_metric_scores(train_scores, test_scores, classifier_names,
                           nb_results, metric_name, file_name, database_name,
                           tag=" vs ".join(labels_names))

        class_file_name = file_name+"-class"
        plot_class_metric_scores(class_test_scores, class_file_name,
                                 labels_names, classifier_names, metric_name)
        logging.info(
            "Done:\t Score graph generation for " + metric_name)
    return results


def publish_all_metrics_scores(iter_results, class_iter_results, directory,
                               data_base_name, stats_iter, label_names,
                               min_size=10):  # pragma: no cover
    results = []
    secure_file_path(os.path.join(directory, "a"))

    for metric_name, scores in iter_results.items():
        if metric_name.endswith("*"):
            metric_name = metric_name[:-1]+"_p"
        train = np.array(scores["mean"].loc["train"])
        test = np.array(scores["mean"].loc["test"])
        classifier_names = np.array(scores["mean"].columns)
        train_std = np.array(scores["std"].loc["train"])
        test_std = np.array(scores["std"].loc["test"])

        file_name = os.path.join(directory, data_base_name + "-mean_on_" + str(
            stats_iter) + "_iter-" + metric_name)
        nb_results = classifier_names.shape[0]

        plot_metric_scores(train, test, classifier_names, nb_results,
                           metric_name, file_name, data_base_name, tag="Averaged",
                           train_STDs=train_std, test_STDs=test_std)
        results += [[classifier_name, metric_name, test_mean, test_std]
                    for classifier_name, test_mean, test_std
                    in zip(classifier_names, test, test_std)]

    for metric_name, scores in class_iter_results.items():
        test = np.array([np.array(scores["mean"].iloc[i, :]) for i in
                         range(scores["mean"].shape[0]) if
                         scores["mean"].iloc[i, :].name[0] == 'test'])
        classifier_names = np.array(scores["mean"].columns)
        test_std = np.array([np.array(scores["std"].iloc[i, :]) for i in
                             range(scores["std"].shape[0]) if
                             scores["std"].iloc[i, :].name[0] == 'test'])

        file_name = os.path.join(directory, data_base_name + "-mean_on_" + str(
            stats_iter) + "_iter-" + metric_name + "-class")

        plot_class_metric_scores(test, file_name, label_names, classifier_names,
                                 metric_name, stds=test_std, tag="averaged")
    return results


def init_plot(results, metric_name, metric_dataframe,
              directory, database_name, class_metric_scores):
    train = np.array(metric_dataframe.loc["train"])
    test = np.array(metric_dataframe.loc["test"])
    class_test = np.array(class_metric_scores.loc["test"])
    classifier_names = np.array(metric_dataframe.columns)

    nb_results = metric_dataframe.shape[1]

    if metric_name.endswith("*"):
        formatted_metric_name = metric_name[:-1]+"_p"
    else:
        formatted_metric_name = metric_name

    file_name = os.path.join(directory, database_name + "-" + formatted_metric_name)

    results += [[classifiers_name, metric_name, test_mean, test_std, class_mean]
                for classifiers_name, test_mean, class_mean, test_std in
                zip(classifier_names, test, np.transpose(class_test),
                    np.zeros(len(test)))]
    return train, test, classifier_names, file_name, nb_results, results, \
        class_test


def plot_metric_scores(train_scores, test_scores, names, nb_results,
                       metric_name,
                       file_name, dataset_name,
                       tag="", train_STDs=None, test_STDs=None,
                       use_plotly=True):  # pragma: no cover
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
    ax.legend((rects[0], rect2[0]), ('Test', 'Train'))
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks(np.arange(nb_results) + barWidth / 2)
    ax.set_xticklabels(names, rotation="vertical")

    try:
        plt.tight_layout()
    except BaseException:
        pass
    f.savefig(file_name + '.png', transparent=True)
    plt.close()
    import pandas as pd
    if train_STDs is None:
        dataframe = pd.DataFrame(np.transpose(np.concatenate((
            train_scores.reshape((train_scores.shape[0], 1)),
            test_scores.reshape((train_scores.shape[0], 1))), axis=1)),
            columns=names, index=["Train", "Test"])
    else:
        dataframe = pd.DataFrame(np.transpose(np.concatenate((
            train_scores.reshape((train_scores.shape[0], 1)),
            train_STDs.reshape((train_scores.shape[0], 1)),
            test_scores.reshape((train_scores.shape[0], 1)),
            test_STDs.reshape((train_scores.shape[0], 1))), axis=1)),
            columns=names, index=["Train", "Train STD", "Test", "Test STD"])
    dataframe.to_csv(file_name + ".csv")
    if use_plotly:
        fig = plotly.graph_objs.Figure()
        fig.add_trace(plotly.graph_objs.Bar(
            name='Train',
            x=names, y=train_scores,
            error_y=dict(type='data', array=train_STDs),
            marker_color="lightgrey",
        ))
        fig.add_trace(plotly.graph_objs.Bar(
            name='Test',
            x=names, y=test_scores,
            error_y=dict(type='data', array=test_STDs),
            marker_color="black",
        ))

        fig.update_layout(
            title="Dataset : {}, metric : {}, task : {} <br> Scores for each classifier <br> Generated on <a href='https://baptiste.bauvin.pages.lis-lab.fr/summit'>SuMMIT</a>.".format(dataset_name, metric_name, tag))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        plotly.offline.plot(fig, filename=file_name + ".html", auto_open=False)
        del fig


def plot_class_metric_scores(class_test_scores, class_file_name,
                             labels_names, classifier_names, metric_name,
                             stds=None, tag=""):  # pragma: no cover
    fig = plotly.graph_objs.Figure()
    for lab_index, scores in enumerate(class_test_scores):
        if stds is None:
            std = None
        else:
            std = stds[lab_index]
        fig.add_trace(plotly.graph_objs.Bar(
            name=labels_names[lab_index],
            x=classifier_names, y=scores,
            error_y=dict(type='data', array=std),
        ))
    fig.update_layout(
        title=metric_name + "<br>" + tag + " scores for each classifier")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    plotly.offline.plot(fig, filename=class_file_name + ".html",
                        auto_open=False)
    del fig


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


def autolabel(rects, ax, set=1, std=None):  # pragma: no cover
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
