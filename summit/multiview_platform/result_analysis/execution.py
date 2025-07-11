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

import pandas as pd

from .duration_analysis import plot_durations, get_duration
from .error_analysis import get_sample_errors, publish_sample_errors, \
    publish_all_sample_errors
from .feature_importances import get_feature_importances, \
    publish_feature_importances
from .metric_analysis import get_metrics_scores, publish_metrics_graphs, \
    publish_all_metrics_scores
from .tracebacks_analysis import save_failed, publish_tracebacks


def analyze(results, stats_iter, benchmark_argument_dictionaries,
            metrics, directory, sample_ids, labels, feature_ids,
            view_names):  # pragma: no cover
    """Used to analyze the results of the previous benchmarks"""
    data_base_name = benchmark_argument_dictionaries[0]["args"]["name"]

    results_means_std, iter_results, flagged_failed, label_names = analyze_iterations(
        results, benchmark_argument_dictionaries,
        stats_iter, metrics, sample_ids, labels, feature_ids, view_names)
    if flagged_failed:
        save_failed(flagged_failed, directory)

    if stats_iter > 1:
        results_means_std = analyze_all(
            iter_results, stats_iter, directory,
            data_base_name, sample_ids, label_names)
    return results_means_std


def analyze_iterations(results, benchmark_argument_dictionaries, stats_iter,
                       metrics, sample_ids, labels, feature_ids, view_names):
    r"""Used to extract and format the results of the different
    experimentations performed.

    Parameters
    ----------
    results : list
        The result list returned by the benchmark execution function. For each
         executed benchmark, contains
        a flag & a result element.
        The flag is a way to identify to which benchmark the results belong,
        formatted this way :
        `flag = iter_index, [classifierPositive, classifierNegative]` with
        - `iter_index` the index of the statistical iteration
        - `[classifierPositive, classifierNegative]` the indices of the labels
        considered positive and negative
        by the classifier (mainly useful for one versus one multiclass
        classification).
    benchmark_argument_dictionaries : list of dicts
        The list of all the arguments passed to the benchmark executing
        functions.
    statsIter : int
        The number of statistical iterations.
    metrics : list of lists
        THe list containing the metrics and their configuration.

    Returns
    -------
    results : list of dicts of dicts
        The list contains a dictionary for each statistical iteration. This
        dictionary contains a dictionary for each
        label combination, regrouping the scores for each metrics and the
        information useful to plot errors on samples.
    """
    logging.info("Start:\t Analyzing all results")
    iter_results = {"metrics_scores": [i for i in range(stats_iter)],
                    "class_metrics_scores": [i for i in range(stats_iter)],
                    "sample_errors": [i for i in range(stats_iter)],
                    "feature_importances": [i for i in range(stats_iter)],
                    "durations": [i for i in range(stats_iter)]}
    flagged_tracebacks_list = []
    fig_errors = []
    for iter_index, result, tracebacks in results:
        arguments = get_arguments(benchmark_argument_dictionaries, iter_index)
        labels_names = list(arguments["labels_dictionary"].values())

        metrics_scores, class_metric_scores = get_metrics_scores(metrics,
                                                                 result,
                                                                 labels_names)
        sample_errors = get_sample_errors(labels, result)
        feature_importances = get_feature_importances(result,
                                                      feature_ids=feature_ids,
                                                      view_names=view_names, )
        durations = get_duration(result)
        directory = arguments["directory"]

        database_name = arguments["args"]["name"]

        flagged_tracebacks_list += publish_tracebacks(directory, database_name,
                                                      labels_names, tracebacks,
                                                      iter_index)
        res = publish_metrics_graphs(metrics_scores, directory, database_name,
                                     labels_names, class_metric_scores)
        publish_sample_errors(sample_errors, directory, database_name,
                              labels_names, sample_ids, labels)
        publish_feature_importances(feature_importances, directory,
                                    database_name, metric_scores=metrics_scores)
        plot_durations(durations, directory, database_name)

        iter_results["metrics_scores"][iter_index] = metrics_scores
        iter_results["class_metrics_scores"][iter_index] = class_metric_scores
        iter_results["sample_errors"][iter_index] = sample_errors
        iter_results["feature_importances"][iter_index] = feature_importances
        iter_results["labels"] = labels
        iter_results["durations"][iter_index] = durations

    logging.info("Done:\t Analyzing all results")

    return res, iter_results, flagged_tracebacks_list, labels_names


def analyze_all(iter_results, stats_iter, directory, data_base_name,
                sample_ids, label_names):  # pragma: no cover
    """Used to format the results in order to plot the mean results on
    the iterations"""
    metrics_analysis, class_metrics_analysis, error_analysis, feature_importances, \
        feature_importances_stds, labels, duration_means, \
        duration_stds = format_previous_results(iter_results)

    results = publish_all_metrics_scores(metrics_analysis,
                                         class_metrics_analysis,
                                         directory,
                                         data_base_name, stats_iter,
                                         label_names)
    publish_all_sample_errors(error_analysis, directory, stats_iter,
                              sample_ids, labels, data_base_name, label_names)
    publish_feature_importances(feature_importances, directory,
                                data_base_name, feature_importances_stds,
                                metric_scores=metrics_analysis)
    plot_durations(duration_means, directory, data_base_name, duration_stds)
    return results


def get_arguments(benchmark_argument_dictionaries, iter_index):
    r"""Used to get the arguments passed to the benchmark executing function
    corresponding to the flag of an
    experimentation.

    Parameters
    ----------
    flag : list
        The needed experimentation's flag.
    benchmark_argument_dictionaries : list of dicts
        The list of all the arguments passed to the benchmark executing
        functions.

    Returns
    -------
    benchmark_argument_dictionary : dict
        All the arguments passed to the benchmark executing function for the
        needed experimentation.
    """
    for benchmark_argument_dictionary in benchmark_argument_dictionaries:
        if benchmark_argument_dictionary["flag"] == iter_index:
            return benchmark_argument_dictionary


def format_previous_results(iter_results_lists):
    """
    Formats each statistical iteration's result into a mean/std analysis for
    the metrics and adds the errors of each statistical iteration.

    Parameters
    ----------
    iter_results_lists : The raw results, for each statistical iteration i
     contains
        - biclass_results[i]["metrics_scores"] is a dictionary with a
        pd.dataframe for each metrics
        - biclass_results[i]["sample_errors"], a dicaitonary with a np.array
        for each classifier.

    Returns
    -------
    metrics_analysis : The mean and std dataframes for each metrics

    error_analysis : A dictionary containing the added errors
                     arrays for each classifier

    """
    metrics_analysis = {}
    class_metrics_analysis = {}
    feature_importances_analysis = {}
    feature_importances_stds = {}

    metric_concat_dict = {}
    for iter_index, metrics_score in enumerate(
            iter_results_lists["metrics_scores"]):
        for metric_name, dataframe in metrics_score.items():
            if metric_name not in metric_concat_dict:
                metric_concat_dict[metric_name] = dataframe
            else:
                metric_concat_dict[metric_name] = pd.concat(
                    [metric_concat_dict[metric_name], dataframe])

    for metric_name, dataframe in metric_concat_dict.items():
        metrics_analysis[metric_name] = {}
        metrics_analysis[metric_name][
            "mean"] = dataframe.groupby(dataframe.index).mean()
        metrics_analysis[metric_name][
            "std"] = dataframe.groupby(dataframe.index).std(ddof=0)

    class_metric_concat_dict = {}
    for iter_index, class_metrics_score in enumerate(
            iter_results_lists["class_metrics_scores"]):
        for metric_name, dataframe in class_metrics_score.items():
            if metric_name not in class_metric_concat_dict:
                class_metric_concat_dict[metric_name] = dataframe
            else:
                class_metric_concat_dict[metric_name] = pd.concat(
                    [class_metric_concat_dict[metric_name], dataframe])

    for metric_name, dataframe in class_metric_concat_dict.items():
        class_metrics_analysis[metric_name] = {}
        class_metrics_analysis[metric_name][
            "mean"] = dataframe.groupby(dataframe.index).mean()
        class_metrics_analysis[metric_name][
            "std"] = dataframe.groupby(dataframe.index).std(ddof=0)

    durations_df_concat = pd.DataFrame(dtype=float)
    for iter_index, durations_df in enumerate(iter_results_lists["durations"]):
        durations_df_concat = pd.concat((durations_df_concat, durations_df),
                                        axis=1)
    durations_df_concat = durations_df_concat.astype(float)
    grouped_df = durations_df_concat.groupby(durations_df_concat.columns,
                                             axis=1)
    duration_means = grouped_df.mean()
    duration_stds = grouped_df.std()

    importance_concat_dict = {}
    for iter_index, view_feature_importances in enumerate(
            iter_results_lists["feature_importances"]):
        for view_name, feature_importances in view_feature_importances.items():
            if view_name not in importance_concat_dict:
                importance_concat_dict[view_name] = feature_importances
            else:
                importance_concat_dict[view_name] = pd.concat(
                    [importance_concat_dict[view_name], feature_importances])

    for view_name, dataframe in importance_concat_dict.items():
        feature_importances_analysis[view_name] = dataframe.groupby(
            dataframe.index).mean()

        feature_importances_stds[view_name] = dataframe.groupby(
            dataframe.index).std(ddof=0)

    added_sample_errors = {}
    for sample_errors in iter_results_lists["sample_errors"]:
        for classifier_name, errors in sample_errors.items():
            if classifier_name not in added_sample_errors:
                added_sample_errors[classifier_name] = errors
            else:
                added_sample_errors[classifier_name] += errors
    error_analysis = added_sample_errors
    return metrics_analysis, class_metrics_analysis, error_analysis, \
        feature_importances_analysis, \
        feature_importances_stds, iter_results_lists["labels"], \
        duration_means, duration_stds
