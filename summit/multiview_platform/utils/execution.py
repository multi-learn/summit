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
import argparse
import logging
import os
import pickle
import time

import numpy as np
import sklearn

from . import get_multiview_db as DB
from ..utils.configuration import save_config


def parse_the_args(arguments):
    """Used to parse the args entered by the user"""

    parser = argparse.ArgumentParser(
        description='This file is used to benchmark the scores fo multiple '
                    'classification algorithm on multiview data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')

    groupStandard = parser.add_argument_group('Standard arguments')
    groupStandard.add_argument('--config_path', metavar='STRING',
                               action='store',
                               help='Path to the hdf5 dataset or database '
                                    'folder (default: %(default)s)',
                               default='../config_files/config.yml')
    args = parser.parse_args(arguments)
    return args


def init_random_state(random_state_arg, directory):
    r"""
    Used to init a random state.
    If no random state is specified, it will generate a 'random' seed.
    If the `randomSateArg` is a string containing only numbers, it will be converted in
     an int to generate a seed.
    If the `randomSateArg` is a string with letters, it must be a path to a pickled random
    state file that will be loaded.
    The function will also pickle the new random state in a file tobe able to retrieve it later.
    Tested


    Parameters
    ----------
    random_state_arg : None or string
        See function description.
    directory : string
        Path to the results directory.

    Returns
    -------
    random_state : numpy.random.RandomState object
        This random state will be used all along the benchmark .
    """

    if random_state_arg is None:
        random_state = np.random.RandomState(random_state_arg)
    else:
        try:
            seed = int(random_state_arg)
            random_state = np.random.RandomState(seed)
        except ValueError:
            file_name = random_state_arg
            with open(file_name, 'rb') as handle:
                random_state = pickle.load(handle)
    with open(os.path.join(directory, "random_state.pickle"), "wb") as handle:
        pickle.dump(random_state, handle)
    return random_state


def init_stats_iter_random_states(stats_iter, random_state):
    r"""
    Used to initialize multiple random states if needed because of multiple statistical iteration of the same benchmark

    Parameters
    ----------
    stats_iter : int
        Number of statistical iterations of the same benchmark done (with a different random state).
    random_state : numpy.random.RandomState object
        The random state of the whole experimentation, that will be used to generate the ones for each
        statistical iteration.

    Returns
    -------
    stats_iter_random_states : list of numpy.random.RandomState objects
        Multiple random states, one for each sattistical iteration of the same benchmark.
    """
    if stats_iter > 1:
        stats_iter_random_states = [
            np.random.RandomState(random_state.randint(5000)) for _ in
            range(stats_iter)]
    else:
        stats_iter_random_states = [random_state]
    return stats_iter_random_states


def get_database_function(name, type_var):
    r"""Used to get the right database extraction function according to the type of database and it's name

    Parameters
    ----------
    name : string
        Name of the database.
    type_var : string
        type of dataset hdf5 or csv

    Returns
    -------
    getDatabase : function
        The function that will be used to extract the database
    """
    if name not in ["fake", "plausible"]:
        get_database = getattr(DB, "get_classic_db_" + type_var[1:])
    else:
        get_database = getattr(DB, "get_" + name + "_db_" + type_var[1:])
    return get_database


def init_log_file(name, views, cl_type, log, debug, label,
                  result_directory, args):
    r"""Used to init the directory where the preds will be stored and the log file.

    First this function will check if the result directory already exists (only one per minute is allowed).

    If the the result directory name is available, it is created, and the logfile is initiated.

    Parameters
    ----------
    name : string
        Name of the database.
    views : list of strings
        List of the view names that will be used in the benchmark.
    cl_type : list of strings
        Type of benchmark that will be made .
    log : bool
        Whether to show the log file in console or hide it.
    debug : bool
        for debug option
    label : str  for label

    result_directory : str name of the result directory

    add_noise : bool for add noise

    noise_std : level of std noise

    Returns
    -------
    results_directory : string
        Reference to the main results directory for the benchmark.
    """
    if views is None:
        views = []
    # result_directory = os.path.join(os.path.dirname(
    #     os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    #                                 result_directory)
    if debug:
        result_directory = os.path.join(result_directory, name,
                                        "debug_started_" + time.strftime(
                                            "%Y_%m_%d-%H_%M_%S") + "_" + label)
    else:
        result_directory = os.path.join(result_directory, name,
                                        "started_" + time.strftime(
                                            "%Y_%m_%d-%H_%M") + "_" + label)
    log_file_name = time.strftime("%Y_%m_%d-%H_%M") + "-" + ''.join(
        cl_type) + "-" + "_".join(views) + "-" + name + "-LOG.log"
    if os.path.exists(result_directory):  # pragma: no cover
        raise NameError("The result dir already exists, wait 1 min and retry")
    log_file_path = os.path.join(result_directory, log_file_name)
    os.makedirs(os.path.dirname(log_file_path))
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        filename=log_file_path, level=logging.INFO,
                        filemode='w')
    if log:
        logging.getLogger().addHandler(logging.StreamHandler())
    save_config(result_directory, args)
    return result_directory


def gen_splits(labels, split_ratio, stats_iter_random_states):
    r"""Used to _gen the train/test splits using one or multiple random states.

    Parameters
    ----------
    labels : numpy.ndarray
        Name of the database.
    split_ratio : float
        The ratio of samples between train and test set.
    stats_iter_random_states : list of numpy.random.RandomState
        The random states for each statistical iteration.

    Returns
    -------
    splits : list of lists of numpy.ndarray
        For each statistical iteration a couple of numpy.ndarrays is stored with the indices for the training set and
        the ones of the testing set.
    """
    indices = np.arange(len(labels))
    splits = []
    for random_state in stats_iter_random_states:
        folds_obj = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,
                                                                   random_state=random_state,
                                                                   test_size=split_ratio)
        folds = folds_obj.split(indices, labels)
        for fold in folds:
            train_fold, test_fold = fold
        train_indices = indices[train_fold]
        test_indices = indices[test_fold]
        splits.append([train_indices, test_indices])

    return splits


def gen_k_folds(stats_iter, nb_folds, stats_iter_random_states):
    r"""Used to generate folds indices for cross validation for each statistical iteration.

    Parameters
    ----------
    stats_iter : integer
        Number of statistical iterations of the benchmark.
    nb_folds : integer
        The number of cross-validation folds for the benchmark.
    stats_iter_random_states : list of numpy.random.RandomState
        The random states for each statistical iteration.

    Returns
    -------
    folds_list : list of list of sklearn.model_selection.StratifiedKFold
        For each statistical iteration a Kfold stratified (keeping the ratio between classes in each fold).
    """
    if stats_iter > 1:
        folds_list = []
        for random_state in stats_iter_random_states:
            folds_list.append(
                sklearn.model_selection.StratifiedKFold(n_splits=nb_folds,
                                                        random_state=random_state,
                                                        shuffle=True))
    else:
        if isinstance(stats_iter_random_states, list):
            stats_iter_random_states = stats_iter_random_states[0]
        folds_list = [sklearn.model_selection.StratifiedKFold(n_splits=nb_folds,
                                                              random_state=stats_iter_random_states,
                                                              shuffle=True)]
    return folds_list


def init_views(dataset_var, arg_views):
    r"""Used to return the views names that will be used by the
    benchmark, their indices and all the views names.

    Parameters
    ----------
    dataset_var : HDF5 dataset file
        The full dataset that wil be used by the benchmark.
    arg_views : list of strings
        The views that will be used by the benchmark (arg).

    Returns
    -------
    views : list of strings
        Names of the views that will be used by the benchmark.
    view_indices : list of ints
        The list of the indices of the view that will be used in the benchmark (according to the dataset).
    all_views : list of strings
        Names of all the available views in the dataset.
    """
    nb_view = dataset_var.nb_view
    if arg_views is not None:
        allowed_views = arg_views
        all_views = [str(dataset_var.get_view_name(view_index))
                     if not isinstance(dataset_var.get_view_name(view_index), bytes)
                     else dataset_var.get_view_name(view_index).decode("utf-8")
                     for view_index in range(nb_view)]
        views = []
        views_indices = []
        for view_index in range(nb_view):
            view_name = dataset_var.get_view_name(view_index)
            if isinstance(view_name, bytes):
                view_name = view_name.decode("utf-8")
            if view_name in allowed_views:
                views.append(view_name)
                views_indices.append(view_index)
    else:
        views = [str(dataset_var.get_view_name(view_index))
                 if not isinstance(dataset_var.get_view_name(view_index), bytes)
                 else dataset_var.get_view_name(view_index).decode("utf-8")
                 for view_index in range(nb_view)]
        views_indices = range(nb_view)
        all_views = views
    return views, views_indices, all_views


def gen_direcorties_names(directory, stats_iter):
    r"""Used to generate the different directories of each iteration if needed.

    Parameters
    ----------
    directory : string
        Path to the results directory.
    statsIter : int
        The number of statistical iterations.

    Returns
    -------
    directories : list of strings
        Paths to each statistical iterations result directory.
    """
    if stats_iter > 1:
        directories = []
        for i in range(stats_iter):
            directories.append(os.path.join(directory, "iter_" + str(i + 1)))
    else:
        directories = [directory]
    return directories


def find_dataset_names(path, type, names):
    """This function goal is to browse the dataset directory and extrats all
     the needed dataset names."""
    package_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    if os.path.isdir(path):
        pass
    elif os.path.isdir(os.path.join(package_path, path)):
        path = os.path.join(package_path, path)
    else:
        raise ValueError("The provided pathf does not exist ({}) SuMMIT checks "
                         "the prefix from where you are running your script ({}) "
                         "and the summit package prefix ({}). "
                         "You may want to try with an absolute path in the "
                         "config file".format(path, os.getcwd(), package_path))
    available_file_names = [file_name.strip().split(".")[0]
                            for file_name in
                            os.listdir(path)
                            if file_name.endswith(type)]
    if names == ["all"]:
        return path, available_file_names
    elif isinstance(names, str):
        return path, [used_name for used_name in available_file_names if
                      names == used_name]
    elif len(names) > 1:
        selected_names = [used_name for used_name in available_file_names if
                          used_name in names]
        if not selected_names:
            raise ValueError(
                "None of the provided dataset names are available. Available datasets are {}".format(
                    available_file_names))
        return path, [used_name for used_name in available_file_names if
                      used_name in names]
    elif names[0] in available_file_names:
        return path, names
    else:
        raise ValueError(
            "The asked dataset ({}) is not available in {}. \n The available ones are {}".format(
                names[0], path, available_file_names))


def gen_argument_dictionaries(labels_dictionary, directories,
                              splits,
                              hyper_param_search, args, k_folds,
                              stats_iter_random_states, metrics,
                              argument_dictionaries,
                              benchmark, views,
                              views_indices, ):  # pragma: no cover
    r"""Used to generate a dictionary for each benchmark.

    One for each label combination (if multiclass), for each statistical iteration, generates an dictionary with
    all necessary information to perform the benchmark

    Parameters
    ----------
    labels_dictionary : dictionary
        Dictionary mapping labels indices to labels names.
    directories : list of strings
        List of the paths to the result directories for each statistical iteration.
    multiclass_labels : list of lists of numpy.ndarray
        For each label couple, for each statistical iteration a triplet of numpy.ndarrays is stored with the
        indices for the biclass training set, the ones for the biclass testing set and the ones for the
        multiclass testing set.
    labels_combinations : list of lists of numpy.ndarray
        Each original couple of different labels.
    indices_multiclass : list of lists of numpy.ndarray
        For each combination, contains a biclass labels numpy.ndarray with the 0/1 labels of combination.
    hyper_param_search : string
        Type of hyper parameter optimization method
    args : parsed args objects
        All the args passed by the user.
    k_folds : list of list of sklearn.model_selection.StratifiedKFold
        For each statistical iteration a Kfold stratified (keeping the ratio between classes in each fold).
    stats_iter_random_states : list of numpy.random.RandomState objects
        Multiple random states, one for each sattistical iteration of the same benchmark.
    metrics : list of lists
        metrics that will be used to evaluate the algorithms performance.
    argument_dictionaries : dictionary
        Dictionary resuming all the specific arguments for the benchmark, oe dictionary for each classifier.
    benchmark : dictionary
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    nb_views : int
        THe number of views used by the benchmark.
    views : list of strings
        List of the names of the used views.
    views_indices : list of ints
        List of indices (according to the dataset) of the used views.

    Returns
    -------
    benchmarkArgumentDictionaries : list of dicts
        All the needed arguments for the benchmarks.

    """
    benchmark_argument_dictionaries = []
    for iter_index, iterRandomState in enumerate(stats_iter_random_states):
        benchmark_argument_dictionary = {
            "labels_dictionary": labels_dictionary,
            "directory": directories[iter_index],
            "classification_indices": splits[iter_index],
            "args": args,
            "k_folds": k_folds[iter_index],
            "random_state": iterRandomState,
            "hyper_param_search": hyper_param_search,
            "metrics": metrics,
            "argument_dictionaries": argument_dictionaries,
            "benchmark": benchmark,
            "views": views,
            "views_indices": views_indices,
            "flag": iter_index}
        benchmark_argument_dictionaries.append(benchmark_argument_dictionary)
    return benchmark_argument_dictionaries
