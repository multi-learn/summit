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
# Import built-in modules
import logging
import os

import matplotlib as mpl
# Import third party modules
import matplotlib.pyplot as plt
import numpy as np
import plotly


# Import own Modules


def get_sample_errors(groud_truth, results):
    r"""Used to get for each classifier and each sample whether the classifier
     has misclassified the sample or not.

    Parameters
    ----------
    ground_truth : numpy array of 0, 1 and -100 (if multiclass)
        The array with the real labels of the samples
    results : list of MonoviewResult and MultiviewResults objects
        A list containing all the resluts for all the mono- & multi-view
        experimentations.

    Returns
    -------
    sample_errors : dict of np.array
        For each classifier, has an entry with a `np.array` over the samples,
         with a 1 if the samples was
        well-classified, a 0 if not and if it's multiclass classification, a
         -100 if the samples was not seen during
        the one versus one classification.
    """
    sample_errors = {}

    for classifier_result in results:
        error_on_samples = np.equal(classifier_result.full_labels_pred,
                                    groud_truth).astype(int)
        unseen_samples = np.where(groud_truth == -100)[0]
        error_on_samples[unseen_samples] = -100
        sample_errors[
            classifier_result.get_classifier_name()] = error_on_samples
    return sample_errors


def publish_sample_errors(sample_errors, directory, database_name,
                          label_names, sample_ids, labels):  # pragma: no cover
    logging.info("Start:\t Label analysis figure generation")

    base_file_name = os.path.join(directory, database_name + "-")

    nb_classifiers, nb_samples, classifiers_names, \
        data_2d, error_on_samples = gen_error_data(sample_errors)

    np.savetxt(base_file_name + "2D_plot_data.csv", data_2d, delimiter=",")
    np.savetxt(base_file_name + "bar_plot_data.csv", error_on_samples,
               delimiter=",")

    plot_2d(data_2d, classifiers_names, nb_classifiers, base_file_name, database_name,
            sample_ids=sample_ids, labels=labels, label_names=label_names)

    plot_errors_bar(error_on_samples, nb_samples,
                    base_file_name, database_name, sample_ids=sample_ids)

    logging.info("Done:\t Label analysis figures generation")


def publish_all_sample_errors(iter_results, directory,
                              stats_iter,
                              sample_ids, labels, data_base_name, label_names):  # pragma: no cover
    logging.info(
        "Start:\t Global label analysis figure generation")

    nb_samples, nb_classifiers, data, \
        error_on_samples, classifier_names = gen_error_data_glob(iter_results,
                                                                 stats_iter)

    np.savetxt(os.path.join(directory, "clf_errors.csv"), data, delimiter=",")
    np.savetxt(os.path.join(directory, "sample_errors.csv"), error_on_samples,
               delimiter=",")

    plot_2d(data, classifier_names, nb_classifiers,
            os.path.join(directory, ""), data_base_name, stats_iter=stats_iter,
            sample_ids=sample_ids, labels=labels, label_names=label_names)
    plot_errors_bar(error_on_samples, nb_samples, os.path.join(directory, ""), data_base_name,
                    sample_ids=sample_ids)

    logging.info(
        "Done:\t Global label analysis figures generation")


def gen_error_data(sample_errors):
    r"""Used to format the error data in order to plot it efficiently. The
    data is saves in a `.csv` file.

    Parameters
    ----------
    sample_errors : dict of dicts of np.arrays
        A dictionary conatining all the useful data. Organized as :
        `sample_errors[<classifier_name>]["error_on_samples"]` is a np.array
        of ints with a
        - 1 if the classifier `<classifier_name>` classifier well the sample,
        - 0 if it fail to classify the sample,
        - -100 if it did not classify the sample (multiclass one versus one).

    Returns
    -------
    nbClassifiers : int
        Number of different classifiers.
    nbExamples : int
        NUmber of samples.
    nbCopies : int
        The number of times the data is copied (classifier wise) in order for
        the figure to be more readable.
    classifiers_names : list of strs
        The names fo the classifiers.
    data : np.array of shape `(nbClassifiers, nbExamples)`
        A matrix with zeros where the classifier failed to classifiy the
        sample, ones where it classified it well
        and -100 if the sample was not classified.
    error_on_samples : np.array of shape `(nbExamples,)`
        An array counting how many classifiers failed to classifiy each
        samples.
    """
    nb_classifiers = len(sample_errors)
    nb_samples = len(list(sample_errors.values())[0])
    classifiers_names = list(sample_errors.keys())

    data_2d = np.zeros((nb_samples, nb_classifiers))
    for classifierIndex, (classifier_name, error_on_samples) in enumerate(
            sample_errors.items()):
        data_2d[:, classifierIndex] = error_on_samples
    error_on_samples = np.sum(data_2d, axis=1) / nb_classifiers
    return nb_classifiers, nb_samples, classifiers_names, data_2d, error_on_samples


def gen_error_data_glob(iter_results, stats_iter):
    nb_samples = next(iter(iter_results.values())).shape[0]
    nb_classifiers = len(iter_results)
    data = np.zeros((nb_samples, nb_classifiers), dtype=int)
    classifier_names = []
    for clf_index, (classifier_name, error_data) in enumerate(
            iter_results.items()):
        data[:, clf_index] = error_data
        classifier_names.append(classifier_name)
    error_on_samples = np.sum(data, axis=1) / (
        nb_classifiers * stats_iter)
    return nb_samples, nb_classifiers, data, error_on_samples, \
        classifier_names


def plot_2d(data, classifiers_names, nb_classifiers, file_name, dataset_name, labels=None,
            stats_iter=1, use_plotly=True, sample_ids=None, label_names=None):  # pragma: no cover
    r"""Used to generate a 2D plot of the errors.

    Parameters
    ----------
    data : np.array of shape `(nbClassifiers, nbExamples)`
        A matrix with zeros where the classifier failed to classifiy the sample, ones where it classified it well
        and -100 if the sample was not classified.
    classifiers_names : list of str
        The names of the classifiers.
    nb_classifiers : int
        The number of classifiers.
    file_name : str
        The name of the file in which the figure will be saved ("error_analysis_2D.png" will be added at the end)
    minSize : int, optinal, default: 10
        The minimum width and height of the figure.
    width_denominator : float, optional, default: 1.0
        To obtain the image width, the number of classifiers will be divided by this number.
    height_denominator : float, optional, default: 1.0
        To obtain the image width, the number of samples will be divided by this number.
    stats_iter : int, optional, default: 1
        The number of statistical iterations realized.

    Returns
    -------
    """
    if label_names is None:
        label_names = [str(lab) for lab in np.sort(np.unique(labels))]
    fig, ax = plt.subplots(nrows=1, ncols=1, )
    label_index_list = np.concatenate([np.where(labels == i)[0] for i in
                                       np.unique(
                                           labels)])
    cmap, norm = iter_cmap(stats_iter)
    cax = plt.imshow(data[np.flip(label_index_list), :], cmap=cmap, norm=norm,
                     aspect='auto')
    plt.title('Errors depending on the classifier')
    ticks = np.arange(0, nb_classifiers, 1)
    tick_labels = classifiers_names
    plt.xticks(ticks, tick_labels, rotation="vertical")
    plt.yticks([], [])
    plt.ylabel("Examples")
    cbar = fig.colorbar(cax, ticks=[-100 * stats_iter / 2, 0, stats_iter])
    cbar.ax.set_yticklabels(['Unseen', 'Always Wrong', 'Always Right'])

    fig.savefig(file_name + "error_analysis_2D.png", bbox_inches="tight",
                transparent=True)
    plt.close()
    # The following part is used to generate an interactive graph.
    if use_plotly:
        hover_text = [[sample_ids[sample_index] + " failed " + str(
            stats_iter - data[
                sample_index, classifier_index]) + " time(s), labelled " + str(
            label_names[int(labels[sample_index])])
            for classifier_index in range(data.shape[1])]
            for sample_index in range(data.shape[0])]
        fig = plotly.graph_objs.Figure()
        fig.add_trace(plotly.graph_objs.Heatmap(
            x=list(classifiers_names),
            y=[sample_ids[label_ind] for label_ind in label_index_list],
            z=data[label_index_list, :],
            text=[hover_text[label_ind] for label_ind in label_index_list],
            hoverinfo=["y", "x", "text"],
            colorscale="Greys",
            colorbar=dict(tickvals=[0, stats_iter],
                          ticktext=["Always Wrong", "Always Right"]),
            reversescale=True), )
        fig.update_yaxes(title_text="Examples", showticklabels=True)
        fig.update_layout(
            title="Dataset : {} <br> Errors for each classifier <br> Generated on <a href='https://baptiste.bauvin.pages.lis-lab.fr/summit'>SuMMIT</a>.".format(
                dataset_name))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showticklabels=True, )
        plotly.offline.plot(fig, filename=file_name + "error_analysis_2D.html",
                            auto_open=False)
        del fig


def plot_errors_bar(error_on_samples, nb_samples, file_name, dataset_name,
                    use_plotly=True, sample_ids=None):  # pragma: no cover
    r"""Used to generate a barplot of the muber of classifiers that failed to classify each samples

    Parameters
    ----------
    error_on_samples : np.array of shape `(nbExamples,)`
        An array counting how many classifiers failed to classifiy each samples.
    classifiers_names : list of str
        The names of the classifiers.
    nb_classifiers : int
        The number of classifiers.
    nb_samples : int
        The number of samples.
    file_name : str
        The name of the file in which the figure will be saved ("error_analysis_2D.png" will be added at the end)

    Returns
    -------
    """
    fig, ax = plt.subplots()
    x = np.arange(nb_samples)
    plt.bar(x, 1 - error_on_samples)
    plt.title("Number of classifiers that failed to classify each sample")
    fig.savefig(file_name + "error_analysis_bar.png", transparent=True)
    plt.close()
    if use_plotly:
        fig = plotly.graph_objs.Figure(
            [plotly.graph_objs.Bar(x=sample_ids, y=1 - error_on_samples)])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          title="Dataset : {} <br> Error % for each sample <br> Generated on <a href='https://baptiste.bauvin.pages.lis-lab.fr/summit'>SuMMIT</a>.".format(
                              dataset_name)
                          )

        plotly.offline.plot(fig, filename=file_name + "error_analysis_bar.html",
                            auto_open=False)


def iter_cmap(statsIter):  # pragma: no cover
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
