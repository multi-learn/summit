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
import pkgutil
import time
import traceback

import matplotlib
import numpy as np

# Import own modules
from . import monoview_classifiers
from . import multiview_classifiers
from .monoview.exec_classif_mono_view import exec_monoview
from .multiview.exec_multiview import exec_multiview
from .result_analysis.execution import analyze_iterations, analyze
from .utils import execution, dataset, configuration
from .utils.dataset import delete_HDF5
from .utils.organization import secure_file_path

matplotlib.use(
    'Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def init_benchmark(cl_type, monoview_algos, multiview_algos):
    r"""Used to create a list of all the algorithm packages names used for the benchmark.

    First this function will check if the benchmark need mono- or/and multiview
    algorithms and adds to the right
    dictionary the asked algorithms. If none is asked by the user, all will be added.

    If the keyword `"Benchmark"` is used, all mono- and multiview algorithms will be added.

    Parameters
    ----------
    cl_type : List of string
        List of types of needed benchmark
    multiview_algos : List of strings
        List of multiview algorithms needed for the benchmark
    monoview_algos : Listof strings
        List of monoview algorithms needed for the benchmark
    args : ParsedArgumentParser args
        All the input args (used to tune the algorithms)

    Returns
    -------
    benchmark : Dictionary of dictionaries
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    """
    benchmark = {"monoview": {}, "multiview": {}}

    if "monoview" in cl_type:
        if monoview_algos == ['all']:  # pragma: no cover
            benchmark["monoview"] = [name for _, name, isPackage in
                                     pkgutil.iter_modules(
                                         monoview_classifiers.__path__)
                                     if not isPackage]

        else:
            benchmark["monoview"] = monoview_algos

    if "multiview" in cl_type:
        if multiview_algos == ["all"]:  # pragma: no cover
            benchmark["multiview"] = [name for _, name, isPackage in
                                      pkgutil.iter_modules(
                                          multiview_classifiers.__path__)
                                      if not isPackage]
        else:
            benchmark["multiview"] = multiview_algos
    return benchmark


def init_argument_dictionaries(benchmark, views_dictionary,
                               nb_class, init_kwargs, hps_method,
                               hps_kwargs):  # pragma: no cover
    argument_dictionaries = {"monoview": [], "multiview": []}
    if benchmark["monoview"]:
        argument_dictionaries["monoview"] = init_monoview_exps(
            benchmark["monoview"],
            views_dictionary,
            nb_class,
            init_kwargs["monoview"], hps_method, hps_kwargs)
    if benchmark["multiview"]:
        argument_dictionaries["multiview"] = init_multiview_exps(
            benchmark["multiview"],
            views_dictionary,
            nb_class,
            init_kwargs["multiview"], hps_method, hps_kwargs)
    return argument_dictionaries


def init_multiview_exps(classifier_names, views_dictionary, nb_class,
                        kwargs_init, hps_method,
                        hps_kwargs):  # pragma: no cover
    multiview_arguments = []
    for classifier_name in classifier_names:
        arguments = get_path_dict(kwargs_init[classifier_name])
        if hps_method == "Grid":
            multiview_arguments += [
                gen_single_multiview_arg_dictionary(classifier_name,
                                                    arguments,
                                                    nb_class,
                                                    {"param_grid": hps_kwargs[
                                                        classifier_name]},
                                                    views_dictionary=views_dictionary)]
        elif hps_method == "Random":
            hps_kwargs = get_random_hps_args(hps_kwargs, classifier_name)
            multiview_arguments += [
                gen_single_multiview_arg_dictionary(classifier_name,
                                                    arguments,
                                                    nb_class,
                                                    hps_kwargs,
                                                    views_dictionary=views_dictionary)]
        elif hps_method == "None":
            multiview_arguments += [
                gen_single_multiview_arg_dictionary(classifier_name,
                                                    arguments,
                                                    nb_class,
                                                    hps_kwargs,
                                                    views_dictionary=views_dictionary)]
        else:
            raise ValueError('At the moment only "None",  "Random" or "Grid" '
                             'are available as hyper-parameter search '
                             'methods, sadly "{}" is not'.format(hps_method)
                             )

    return multiview_arguments


def init_monoview_exps(classifier_names,
                       views_dictionary, nb_class, kwargs_init, hps_method,
                       hps_kwargs):  # pragma: no cover
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
    views_dictionary : dictionary
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
        for classifier_name in classifier_names:
            if hps_method == "Grid":
                arguments = gen_single_monoview_arg_dictionary(classifier_name,
                                                               kwargs_init,
                                                               nb_class,
                                                               view_index,
                                                               view_name,
                                                               {"param_grid":
                                                                hps_kwargs[
                                                                    classifier_name]})
            elif hps_method == "Random":
                hps_kwargs = get_random_hps_args(hps_kwargs, classifier_name)
                arguments = gen_single_monoview_arg_dictionary(classifier_name,
                                                               kwargs_init,
                                                               nb_class,
                                                               view_index,
                                                               view_name,
                                                               hps_kwargs)
            elif hps_method == "None":
                arguments = gen_single_monoview_arg_dictionary(classifier_name,
                                                               kwargs_init,
                                                               nb_class,
                                                               view_index,
                                                               view_name,
                                                               hps_kwargs)

            else:
                raise ValueError(
                    'At the moment only "None",  "Random" or "Grid" '
                    'are available as hyper-parameter search '
                    'methods, sadly "{}" is not'.format(hps_method)
                )
            monoview_arguments.append(arguments)
    return monoview_arguments


def get_random_hps_args(hps_args, classifier_name):
    hps_dict = {}
    for key, value in hps_args.items():
        if key in ["n_iter", "equivalent_draws"]:
            hps_dict[key] = value
        if key==classifier_name:
            hps_dict["param_distributions"] = value
    return hps_dict


def gen_single_monoview_arg_dictionary(classifier_name, arguments, nb_class,
                                       view_index, view_name, hps_kwargs):
    if classifier_name in arguments:
        classifier_config = dict((key, value) for key, value in arguments[
            classifier_name].items())
    else:
        classifier_config = {}
    return {classifier_name: classifier_config,
            "view_name": view_name,
            "view_index": view_index,
            "classifier_name": classifier_name,
            "nb_class": nb_class,
            "hps_kwargs": hps_kwargs}


def gen_single_multiview_arg_dictionary(classifier_name, arguments, nb_class,
                                        hps_kwargs, views_dictionary=None):
    return {"classifier_name": classifier_name,
            "view_names": list(views_dictionary.keys()),
            'view_indices': list(views_dictionary.values()),
            "nb_class": nb_class,
            "labels_names": None,
            "hps_kwargs": hps_kwargs,
            classifier_name: extract_dict(arguments)
            }


def extract_dict(classifier_config):
    """Reverse function of get_path_dict"""
    extracted_dict = {}
    for key, value in classifier_config.items():
        extracted_dict = set_element(extracted_dict, key, value)
    return extracted_dict


def set_element(dictionary, path, value):
    """Set value in dictionary at the location indicated by path"""
    existing_keys = path.split(".")[:-1]
    dict_state = dictionary
    for existing_key in existing_keys:
        if existing_key in dict_state:
            dict_state = dict_state[existing_key]
        else:
            dict_state[existing_key] = {}
            dict_state = dict_state[existing_key]
    dict_state[path.split(".")[-1]] = value
    return dictionary


def get_path_dict(multiview_classifier_args):
    """This function is used to generate a dictionary with each key being
    the path to the value.
    If given {"key1":{"key1_1":value1}, "key2":value2}, it will return
    {"key1.key1_1":value1, "key2":value2}"""
    path_dict = dict(
        (key, value) for key, value in multiview_classifier_args.items())
    paths = is_dict_in(path_dict)
    while paths:
        for path in paths:
            for key, value in path_dict[path].items():
                path_dict[".".join([path, key])] = value
            path_dict.pop(path)
        paths = is_dict_in(path_dict)
    return path_dict


def is_dict_in(dictionary):
    """
    Returns True if any of the dictionary value is a dictionary itself.

    Parameters
    ----------
    dictionary

    Returns
    -------

    """
    paths = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            paths.append(key)
    return paths


def init_kwargs(args, classifiers_names, framework="monoview"):
    r"""Used to init kwargs thanks to a function in each monoview classifier package.

    Parameters
    ----------
    args : parsed args objects
        All the args passed by the user.
    classifiers_names : list of strings
        List of the benchmarks's monoview classifiers names.

    Returns
    -------
    kwargs : Dictionary
        Dictionary resuming all the specific arguments for the benchmark, one dictionary for each classifier.

        For example, for Adaboost, the KWARGS will be `{"n_estimators":<value>, "base_estimator":<value>}`"""

    logging.info("Start:\t Initializing monoview classifiers arguments")
    kwargs = {}
    for classifiers_name in classifiers_names:
        try:
            if framework == "monoview":
                getattr(monoview_classifiers, classifiers_name)
            else:
                getattr(multiview_classifiers, classifiers_name)
        except AttributeError:
            raise AttributeError(
                classifiers_name + " is not implemented in monoview_classifiers, "
                                   "please specify the name of the file in monoview_classifiers")
        if classifiers_name in args:
            kwargs[classifiers_name] = args[classifiers_name]
        else:
            kwargs[classifiers_name] = {}
    logging.info("Done:\t Initializing monoview classifiers arguments")

    return kwargs


def init_kwargs_func(args, benchmark):
    """
    Dispached the kwargs initialization to monoview and multiview and creates
    the kwargs variable

    Parameters
    ----------
    args : parsed args objects
        All the args passed by the user.

    benchmark : dict
        The name of the mono- and mutli-view classifiers to run in the benchmark

    Returns
    -------

    kwargs : dict
        The arguments for each mono- and multiview algorithms
    """
    monoview_kwargs = init_kwargs(args, benchmark["monoview"],
                                  framework="monoview")
    multiview_kwargs = init_kwargs(args, benchmark["multiview"],
                                   framework="multiview")
    kwargs = {"monoview": monoview_kwargs, "multiview": multiview_kwargs}
    return kwargs


def arange_metrics(metrics, metric_princ):
    """Used to get the metrics list in the right order so that
    the first one is the principal metric specified in args

    Parameters
    ----------
    metrics : dict
        The metrics that will be used in the benchmark

    metric_princ : str
        The name of the metric that need to be used for the hyper-parameter
        optimization process

    Returns
    -------
    metrics : list of lists
        The metrics list, but arranged  so the first one is the principal one."""
    if metric_princ in metrics:
        metrics = dict(
            (key, value) if not key == metric_princ else (key + "*", value) for
            key, value in metrics.items())
    else:
        raise ValueError("{} not in metric pool ({})".format(metric_princ,
                                                             metrics))
    return metrics


def benchmark_init(directory, classification_indices, labels, labels_dictionary,
                   k_folds, dataset_var):
    """
    Initializes the benchmark, by saving the indices of the train
    samples and the cross validation folds.

    Parameters
    ----------
    directory : str
        The benchmark's result directory

    classification_indices : numpy array
        The indices of the samples, splitted for the train/test split

    labels : numpy array
        The labels of the dataset

    labels_dictionary : dict
        The dictionary with labels as keys and their names as values

    k_folds : sklearn.model_selection.Folds object
        The folds for the cross validation process

    Returns
    -------

    """
    logging.info("Start:\t Benchmark initialization")
    secure_file_path(os.path.join(directory, "train_labels.csv"))
    train_indices = classification_indices[0]
    train_labels = dataset_var.get_labels(sample_indices=train_indices)
    np.savetxt(os.path.join(directory, "train_labels.csv"), train_labels,
               delimiter=",")
    np.savetxt(os.path.join(directory, "train_indices.csv"),
               classification_indices[0],
               delimiter=",")
    results_monoview = []
    folds = k_folds.split(np.arange(len(train_labels)), train_labels)
    min_fold_len = int(len(train_labels) / k_folds.n_splits)
    for fold_index, (train_cv_indices, test_cv_indices) in enumerate(folds):
        file_name = os.path.join(directory, "folds", "test_labels_fold_" + str(
            fold_index) + ".csv")
        secure_file_path(file_name)
        np.savetxt(file_name, train_labels[test_cv_indices[:min_fold_len]],
                   delimiter=",")
    labels_names = list(labels_dictionary.values())
    logging.info("Done:\t Benchmark initialization")
    return results_monoview, labels_names


def exec_one_benchmark_mono_core(dataset_var=None, labels_dictionary=None,
                                 directory=None, classification_indices=None,
                                 args=None,
                                 k_folds=None, random_state=None,
                                 hyper_param_search=None, metrics=None,
                                 argument_dictionaries=None,
                                 benchmark=None, views=None, views_indices=None,
                                 flag=None, labels=None,
                                 track_tracebacks=False, nb_cores=1):  # pragma: no cover

    results_monoview, labels_names = benchmark_init(directory,
                                                    classification_indices,
                                                    labels,
                                                    labels_dictionary, k_folds,
                                                    dataset_var)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info("Start:\t monoview benchmark")
    traceback_outputs = {}
    for arguments in argument_dictionaries["monoview"]:
        try:
            X = dataset_var.get_v(arguments["view_index"])
            Y = dataset_var.get_labels()
            results_monoview += [
                exec_monoview(directory, X, Y, args["name"], labels_names,
                              classification_indices, k_folds,
                              nb_cores, args["file_type"], args["pathf"], random_state,
                              hyper_param_search=hyper_param_search,
                              metrics=metrics, feature_ids=dataset_var.feature_ids[arguments["view_index"]],
                              **arguments)]
        except BaseException:
            if track_tracebacks:
                traceback_outputs[
                    arguments["classifier_name"] + "-" + arguments[
                        "view_name"]] = traceback.format_exc()
            else:
                raise
    logging.info("Done:\t monoview benchmark")

    logging.info("Start:\t multiview benchmark")
    results_multiview = []
    for arguments in argument_dictionaries["multiview"]:
        try:
            results_multiview += [
                exec_multiview(directory, dataset_var, args["name"],
                               classification_indices,
                               k_folds, nb_cores, args["file_type"],
                               args["pathf"], labels_dictionary, random_state,
                               labels,
                               hps_method=hyper_param_search,
                               metrics=metrics, n_iter=args["hps_iter"],
                               **arguments)]
        except BaseException:
            if track_tracebacks:
                traceback_outputs[
                    arguments["classifier_name"]] = traceback.format_exc()
            else:
                raise
    logging.info("Done:\t multiview benchmark")

    return [flag, results_monoview + results_multiview, traceback_outputs]


def exec_benchmark(nb_cores, stats_iter,
                   benchmark_arguments_dictionaries,
                   directory, metrics, dataset_var, track_tracebacks,
                   exec_one_benchmark_mono_core=exec_one_benchmark_mono_core,
                   analyze=analyze, delete=delete_HDF5,
                   analyze_iterations=analyze_iterations):  # pragma: no cover
    r"""Used to execute the needed benchmark(s) on multicore or mono-core functions.

    Parameters
    ----------
    nb_cores : int
        Number of threads that the benchmarks can use.
    stats_iter : int
        Number of statistical iterations that have to be done.
    benchmark_arguments_dictionaries : list of dictionaries
        All the needed arguments for the benchmarks.
    classification_indices : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.
    directories : list of strings
        List of the paths to the result directories for each statistical iteration.
    directory : string
        Path to the main results directory.
    multi_class_labels : ist of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.
    metrics : list of lists
        metrics that will be used to evaluate the algorithms performance.
    labels_dictionary : dictionary
        Dictionary mapping labels indices to labels names.
    nb_labels : int
        Total number of different labels in the dataset.
    dataset_var : HDF5 dataset file
        The full dataset that wil be used by the benchmark.
    classifiers_names : list of strings
        List of the benchmarks's monoview classifiers names.
    rest_of_the_args :
        Just used for testing purposes


    Returns
    -------
    results : list of lists
        The results of the benchmark.
    """
    logging.info("Start:\t Executing all the needed benchmarks")
    results = []
    for arguments in benchmark_arguments_dictionaries:
        benchmark_results = exec_one_benchmark_mono_core(
            dataset_var=dataset_var,
            track_tracebacks=track_tracebacks, nb_cores=nb_cores,
            **arguments)
        analyze_iterations([benchmark_results],
                           benchmark_arguments_dictionaries, stats_iter,
                           metrics, sample_ids=dataset_var.sample_ids,
                           labels=dataset_var.get_labels(),
                           feature_ids=dataset_var.feature_ids,
                           view_names=dataset_var.view_names)
        results += [benchmark_results]
    logging.info("Done:\t Executing all the needed benchmarks")

    # Do everything with flagging
    logging.info("Start:\t Analyzing predictions")
    results_mean_stds = analyze(results, stats_iter,
                                benchmark_arguments_dictionaries,
                                metrics,
                                directory,
                                dataset_var.sample_ids,
                                dataset_var.get_labels(),dataset_var.feature_ids,
                                dataset_var.view_names)
    logging.info("Done:\t Analyzing predictions")
    return results_mean_stds


def exec_classif(arguments):  # pragma: no cover
    """
    Runs the benchmark with the given arguments

    Parameters
    ----------
    arguments :

    Returns
    -------


    >>> exec_classif([--config_path, /path/to/config/files/])
    >>>
    """
    start = time.time()
    args = execution.parse_the_args(arguments)
    args = configuration.get_the_args(args.config_path)
    import sys
    if not sys.platform in ["win32", "cygwin"]:
        os.nice(args["nice"])
    nb_cores = args["nb_cores"]
    if nb_cores == 1:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    stats_iter = args["stats_iter"]
    hps_method = args["hps_type"]
    hps_kwargs = args["hps_args"]
    cl_type = args["type"]
    monoview_algos = args["algos_monoview"]
    multiview_algos = args["algos_multiview"]
    path, dataset_list = execution.find_dataset_names(args["pathf"],
                                                      args["file_type"],
                                                      args["name"])
    args["pathf"] = path
    for dataset_name in dataset_list:
        # noise_results = []
        # for noise_std in args["noise_std"]:

        directory = execution.init_log_file(dataset_name, args["views"],
                                            args["file_type"],
                                            args["log"], args["debug"],
                                            args["label"],
                                            args["res_dir"],
                                            args)

        random_state = execution.init_random_state(args["random_state"],
                                                   directory)
        stats_iter_random_states = execution.init_stats_iter_random_states(
            stats_iter,
            random_state)

        get_database = execution.get_database_function(dataset_name,
                                                       args["file_type"])

        dataset_var, labels_dictionary, datasetname = get_database(
            args["views"],
            args["pathf"], dataset_name,
            args["nb_class"],
            args["classes"],
            random_state,
            args["full"],
        )
        args["name"] = datasetname
        splits = execution.gen_splits(dataset_var.get_labels(),
                                      args["split"],
                                      stats_iter_random_states)

        k_folds = execution.gen_k_folds(stats_iter, args["nb_folds"],
                                        stats_iter_random_states)


        views, views_indices, all_views = execution.init_views(dataset_var,
                                                               args[
                                                                   "views"])
        views_dictionary = dataset_var.get_view_dict()
        nb_views = len(views)
        nb_class = dataset_var.get_nb_class()

        metrics = args["metrics"]
        if metrics == "all":
            metrics_names = [name for _, name, isPackage
                             in pkgutil.iter_modules(
                                 [os.path.join(os.path.dirname(
                                     os.path.dirname(os.path.realpath(__file__))),
                                     'metrics')]) if
                             not isPackage and name not in ["framework",
                                                            "log_loss",
                                                            "matthews_corrcoef",
                                                            "roc_auc_score"]]
            metrics = dict((metric_name, {})
                           for metric_name in metrics_names)
        metrics = arange_metrics(metrics, args["metric_princ"])

        benchmark = init_benchmark(cl_type, monoview_algos, multiview_algos, )
        init_kwargs = init_kwargs_func(args, benchmark)
        data_base_time = time.time() - start
        argument_dictionaries = init_argument_dictionaries(
            benchmark, views_dictionary,
            nb_class, init_kwargs, hps_method, hps_kwargs)
        # argument_dictionaries = initMonoviewExps(benchmark, viewsDictionary,
        #                                         NB_CLASS, initKWARGS)
        directories = execution.gen_direcorties_names(directory, stats_iter)
        benchmark_argument_dictionaries = execution.gen_argument_dictionaries(
            labels_dictionary, directories,
            splits,
            hps_method, args, k_folds,
            stats_iter_random_states, metrics,
            argument_dictionaries, benchmark,
            views, views_indices)
        exec_benchmark(nb_cores, stats_iter, benchmark_argument_dictionaries,
                       directory, metrics, dataset_var, args["track_tracebacks"])
