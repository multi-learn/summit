import errno
import logging
import math
import os
import pkgutil
import time
import traceback

import matplotlib
import itertools
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

# Import own modules
from . import monoview_classifiers
from . import multiview_classifiers
from .multiview.exec_multiview import exec_multiview, exec_multiview_multicore
from .monoview.exec_classif_mono_view import exec_monoview, exec_monoview_multicore
from .utils.dataset import delete_HDF5
from .result_analysis import get_results, plot_results_noise, analyze_iterations
from .utils import execution, dataset, multiclass, configuration

matplotlib.use(
    'Agg')  # Anti-Grain Geometry C++ library to make a raster (pixel) image of the figure



# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def init_benchmark(cl_type, monoview_algos, multiview_algos, args):
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
        if monoview_algos == ['all']:
            benchmark["monoview"] = [name for _, name, isPackage in
                                     pkgutil.iter_modules(monoview_classifiers.__path__)
                                     if not isPackage]

        else:
            benchmark["monoview"] = monoview_algos

    if "multiview" in cl_type:
        if multiview_algos==["all"]:
            benchmark["multiview"] = [name for _, name, isPackage in
                                     pkgutil.iter_modules(multiview_classifiers.__path__)
                                     if not isPackage]
        else:
            benchmark["multiview"] = multiview_algos
    return benchmark


def init_argument_dictionaries(benchmark, views_dictionary,
                                nb_class, init_kwargs):
    argument_dictionaries = {"monoview": [], "multiview": []}
    if benchmark["monoview"]:
        argument_dictionaries["monoview"] = init_monoview_exps(
                                                   benchmark["monoview"],
                                                   views_dictionary,
                                                   nb_class,
                                                   init_kwargs["monoview"])
    if benchmark["multiview"]:
        argument_dictionaries["multiview"] = init_multiview_exps(benchmark["multiview"],
                                                   views_dictionary,
                                                   nb_class,
                                                   init_kwargs["multiview"])
    return argument_dictionaries


def init_multiview_exps(classifier_names, views_dictionary, nb_class, kwargs_init):
    multiview_arguments = []
    for classifier_name in classifier_names:
        if multiple_args(get_path_dict(kwargs_init[classifier_name])):
            multiview_arguments += gen_multiple_args_dictionnaries(
                                                                  nb_class,
                                                                  kwargs_init,
                                                                  classifier_name,
                                                                  views_dictionary=views_dictionary,
                                                                  framework="multiview")
        else:
            arguments = get_path_dict(kwargs_init[classifier_name])
            multiview_arguments += [gen_single_multiview_arg_dictionary(classifier_name,
                                                                        arguments,
                                                                        nb_class,
                                                                        views_dictionary=views_dictionary)]
    return multiview_arguments


def init_monoview_exps(classifier_names,
                       views_dictionary, nb_class, kwargs_init):
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
        for classifier in classifier_names:
            if multiple_args(kwargs_init[classifier]):
                monoview_arguments += gen_multiple_args_dictionnaries(nb_class,
                                                                      kwargs_init,
                                                                      classifier,
                                                                      view_name,
                                                                      view_index)
            else:
                arguments = gen_single_monoview_arg_dictionary(classifier,
                                                               kwargs_init,
                                                               nb_class,
                                                               view_index,
                                                               view_name)
                monoview_arguments.append(arguments)
    return monoview_arguments


def gen_single_monoview_arg_dictionary(classifier_name, arguments, nb_class,
                                       view_index, view_name):
    if classifier_name in arguments:
        classifier_config = dict((key, value[0]) for key, value in arguments[
                            classifier_name].items())
    else:
        classifier_config = {}
    return {classifier_name: classifier_config,
            "view_name": view_name,
            "view_index": view_index,
            "classifier_name": classifier_name,
            "nb_class": nb_class}


def gen_single_multiview_arg_dictionary(classifier_name,arguments,nb_class,
                                        views_dictionary=None):
    return {"classifier_name": classifier_name,
            "view_names": list(views_dictionary.keys()),
            'view_indices': list(views_dictionary.values()),
            "nb_class": nb_class,
            "labels_names": None,
            classifier_name: extract_dict(arguments)
            }


def extract_dict(classifier_config):
    """Reverse function of get_path_dict"""
    extracted_dict = {}
    for key, value in classifier_config.items():
        if isinstance(value, list):
            extracted_dict = set_element(extracted_dict, key, value[0])
        else:
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


def multiple_args(classifier_configuration):
    """Checks if multiple values were provided for at least one arg"""
    listed_args = [type(value) == list and len(value)>1 for key, value in
                   classifier_configuration.items()]
    if True in listed_args:
        return True
    else: 
        return False


def get_path_dict(multiview_classifier_args):
    """This function is used to generate a dictionary with each key being
    the path to the value.
    If given {"key1":{"key1_1":value1}, "key2":value2}, it will return
    {"key1.key1_1":value1, "key2":value2}"""
    path_dict = dict((key, value) for key, value in multiview_classifier_args.items())
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


def gen_multiple_kwargs_combinations(cl_kwrags):
    """
    Generates all the possible combination of the asked args

    Parameters
    ----------
    cl_kwrags : dict
        The arguments, with one at least having multiple values

    Returns
    -------
    kwargs_combination : list
        The list of all the combinations of arguments

    reduced_kwargs_combination : list
        The reduced names and values of the arguments will be used in the naming
        process of the different classifiers

    """
    values = list(cl_kwrags.values())
    listed_values = [[_] if type(_) is not list else _ for _ in values]
    values_cartesian_prod = [_ for _ in itertools.product(*listed_values)]
    keys = cl_kwrags.keys()
    kwargs_combination = [dict((key, value) for key, value in zip(keys, values))
                          for values in values_cartesian_prod]

    reduce_dict = {DecisionTreeClassifier: "DT", }
    reduced_listed_values = [
        [_ if type(_) not in reduce_dict else reduce_dict[type(_)] for _ in
         list_] for list_ in listed_values]
    reduced_values_cartesian_prod = [_ for _ in itertools.product(*reduced_listed_values)]
    reduced_kwargs_combination = [dict((key, value) for key, value in zip(keys, values))
                          for values in reduced_values_cartesian_prod]
    return kwargs_combination, reduced_kwargs_combination


def gen_multiple_args_dictionnaries(nb_class, kwargs_init, classifier,
                                    view_name=None, view_index=None,
                                    views_dictionary=None,
                                    framework="monoview"):
    """
    Used in the case of mutliple arguments asked in the config file.
    Will combine the arguments to explore all the possibilities.

    Parameters
    ----------
    nb_class : int,
        The number of classes in the dataset

    kwargs_init : dict
        The arguments given in the config file

    classifier : str
        The name of the classifier for which multiple arguments have been asked

    view_name : str
        The name of the view in consideration.

    view_index : int
        The index of the view in consideration

    views_dictionary : dict
        The dictionary of all the views indices and their names

    framework : str
        Either monoview or multiview

    Returns
    -------
    args_dictionaries : list
        The list of all the possible combination of asked arguments

    """
    if framework=="multiview":
        classifier_config = get_path_dict(kwargs_init[classifier])
    else:
        classifier_config = kwargs_init[classifier]
    multiple_kwargs_list, reduced_multiple_kwargs_list = gen_multiple_kwargs_combinations(classifier_config)
    multiple_kwargs_dict = dict(
        (classifier+"_"+"_".join(map(str,list(reduced_dictionary.values()))), dictionary)
        for reduced_dictionary, dictionary in zip(reduced_multiple_kwargs_list, multiple_kwargs_list ))
    args_dictionnaries = [gen_single_monoview_arg_dictionary(classifier_name,
                                                             arguments,
                                                             nb_class,
                                                             view_index=view_index,
                                                             view_name=view_name)
                           if framework=="monoview" else
                           gen_single_multiview_arg_dictionary(classifier_name,
                                                            arguments,
                                                            nb_class,
                                                            views_dictionary=views_dictionary)
                           for classifier_name, arguments
                           in multiple_kwargs_dict.items()]
    return args_dictionnaries


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

    logging.debug("Start:\t Initializing monoview classifiers arguments")
    kwargs = {}
    for classifiers_name in classifiers_names:
        try:
            if framework=="monoview":
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
    logging.debug("Done:\t Initializing monoview classifiers arguments")

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
    monoview_kwargs = init_kwargs(args, benchmark["monoview"], framework="monoview")
    multiview_kwargs = init_kwargs(args, benchmark["multiview"], framework="multiview")
    kwargs = {"monoview":monoview_kwargs, "multiview":multiview_kwargs}
    return kwargs


# def init_multiview_kwargs(args, classifiers_names):
#     logging.debug("Start:\t Initializing multiview classifiers arguments")
#     multiview_kwargs = {}
#     for classifiers_name in classifiers_names:
#         try:
#             getattr(multiview_classifiers, classifiers_name)
#         except AttributeError:
#             raise AttributeError(
#                 classifiers_name + " is not implemented in mutliview_classifiers, "
#                                   "please specify the name of the coressponding .py "
#                                    "file in mutliview_classifiers")
#         multiview_kwargs[classifiers_name] = args[classifiers_name]
#     logging.debug("Done:\t Initializing multiview classifiers arguments")
#     return multiview_kwargs


# def init_multiview_arguments(args, benchmark, views, views_indices,
#                              argument_dictionaries, random_state, directory,
#                              results_monoview, classification_indices):
#     """Used to add each monoview exeperience args to the list of monoview experiences args"""
#     logging.debug("Start:\t Initializing multiview classifiers arguments")
#     multiview_arguments = []
#     if "multiview" in benchmark:
#         for multiview_algo_name in benchmark["multiview"]:
#             mutliview_module = getattr(multiview_classifiers,
#                                       multiview_algo_name)
#
#             multiview_arguments += mutliview_module.getArgs(args, benchmark,
#                                                           views, views_indices,
#                                                           random_state,
#                                                           directory,
#                                                           results_monoview,
#                                                           classification_indices)
#     argument_dictionaries["multiview"] = multiview_arguments
#     logging.debug("Start:\t Initializing multiview classifiers arguments")
#     return argument_dictionaries


def arange_metrics(metrics, metric_princ):
    """Used to get the metrics list in the right order so that
    the first one is the principal metric specified in args

    Parameters
    ----------
    metrics : list of lists
        The metrics that will be used in the benchmark

    metric_princ : str
        The name of the metric that need to be used for the hyper-parameter
        optimization process

    Returns
    -------
    metrics : list of lists
        The metrics list, but arranged  so the first one is the principal one."""
    if [metric_princ] in metrics:
        metric_index = metrics.index([metric_princ])
        first_metric = metrics[0]
        metrics[0] = [metric_princ]
        metrics[metric_index] = first_metric
    else:
        raise AttributeError(metric_princ + " not in metric pool")
    return metrics


def benchmark_init(directory, classification_indices, labels, labels_dictionary,
                   k_folds, dataset_var):
    """
    Initializes the benchmark, by saving the indices of the train
    examples and the cross validation folds.

    Parameters
    ----------
    directory : str
        The benchmark's result directory

    classification_indices : numpy array
        The indices of the examples, splitted for the train/test split

    labels : numpy array
        The labels of the dataset

    labels_dictionary : dict
        The dictionary with labels as keys and their names as values

    k_folds : sklearn.model_selection.Folds object
        The folds for the cross validation process

    Returns
    -------

    """
    logging.debug("Start:\t Benchmark initialization")
    if not os.path.exists(os.path.dirname(os.path.join(directory, "train_labels.csv"))):
        try:
            os.makedirs(os.path.dirname(os.path.join(directory, "train_labels.csv")))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    train_indices = classification_indices[0]
    train_labels = dataset_var.get_labels(example_indices=train_indices)
    np.savetxt(os.path.join(directory, "train_labels.csv"), train_labels, delimiter=",")
    np.savetxt(os.path.join(directory, "train_indices.csv"), classification_indices[0],
               delimiter=",")
    results_monoview = []
    folds = k_folds.split(np.arange(len(train_labels)), train_labels)
    min_fold_len = int(len(train_labels) / k_folds.n_splits)
    for fold_index, (train_cv_indices, test_cv_indices) in enumerate(folds):
        file_name = os.path.join(directory, "folds", "test_labels_fold_" + str(
            fold_index) + ".csv")
        if not os.path.exists(os.path.dirname(file_name)):
            try:
                os.makedirs(os.path.dirname(file_name))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.savetxt(file_name, train_labels[test_cv_indices[:min_fold_len]],
                   delimiter=",")
    labels_names = list(labels_dictionary.values())
    logging.debug("Done:\t Benchmark initialization")
    return results_monoview, labels_names


# def exec_one_benchmark(core_index=-1, labels_dictionary=None, directory=None,
#                      classification_indices=None, args=None,
#                      k_folds=None, random_state=None, hyper_param_search=None,
#                      metrics=None, argument_dictionaries=None,
#                      benchmark=None, views=None, views_indices=None, flag=None,
#                      labels=None,
#                      exec_monoview_multicore=exec_monoview_multicore,
#                      exec_multiview_multicore=exec_multiview_multicore,):
#     """Used to run a benchmark using one core. ExecMonoview_multicore, initMultiviewArguments and
#      exec_multiview_multicore args are only used for tests"""
#
#     results_monoview, labels_names = benchmark_init(directory,
#                                                     classification_indices, labels,
#                                                     labels_dictionary, k_folds)
#
#     logging.debug("Start:\t monoview benchmark")
#     results_monoview += [
#         exec_monoview_multicore(directory, args["name"], labels_names,
#                                classification_indices, k_folds,
#                                core_index, args["file_type"], args["pathf"], random_state,
#                                labels,
#                                hyper_param_search=hyper_param_search,
#                                metrics=metrics,
#                                n_iter=args["hps_iter"], **argument)
#         for argument in argument_dictionaries["Monoview"]]
#     logging.debug("Done:\t monoview benchmark")
#
#
#     logging.debug("Start:\t multiview benchmark")
#     results_multiview = [
#         exec_multiview_multicore(directory, core_index, args["name"],
#                                 classification_indices, k_folds, args["file_type"],
#                                 args["pathf"], labels_dictionary, random_state,
#                                 labels, hyper_param_search=hyper_param_search,
#                                 metrics=metrics, n_iter=args["hps_iter"],
#                                 **arguments)
#         for arguments in argument_dictionaries["multiview"]]
#     logging.debug("Done:\t multiview benchmark")
#
#     return [flag, results_monoview + results_multiview]
#
#
# def exec_one_benchmark_multicore(nb_cores=-1, labels_dictionary=None,
#                                  directory=None, classification_indices=None,
#                                  args=None,
#                                  k_folds=None, random_state=None,
#                                  hyper_param_search=None, metrics=None,
#                                  argument_dictionaries=None,
#                                  benchmark=None, views=None, views_indices=None,
#                                  flag=None, labels=None,
#                                  exec_monoview_multicore=exec_monoview_multicore,
#                                  exec_multiview_multicore=exec_multiview_multicore,):
#     """Used to run a benchmark using multiple cores. ExecMonoview_multicore, initMultiviewArguments and
#      exec_multiview_multicore args are only used for tests"""
#
#     results_monoview, labels_names = benchmark_init(directory,
#                                                     classification_indices, labels,
#                                                     labels_dictionary, k_folds)
#
#     logging.debug("Start:\t monoview benchmark")
#     nb_experiments = len(argument_dictionaries["monoview"])
#     nb_multicore_to_do = int(math.ceil(float(nb_experiments) / nb_cores))
#     for step_index in range(nb_multicore_to_do):
#         results_monoview += (Parallel(n_jobs=nb_cores)(
#             delayed(exec_monoview_multicore)(directory, args["name"], labels_names,
#                                             classification_indices, k_folds,
#                                             core_index, args["file_type"], args["pathf"],
#                                             random_state, labels,
#                                             hyper_param_search=hyper_param_search,
#                                             metrics=metrics,
#                                             n_iter=args["hps_iter"],
#                                             **argument_dictionaries["monoview"][
#                                             core_index + step_index * nb_cores])
#             for core_index in
#             range(min(nb_cores, nb_experiments - step_index * nb_cores))))
#     logging.debug("Done:\t monoview benchmark")
#
#     logging.debug("Start:\t multiview arguments initialization")
#     # argument_dictionaries = initMultiviewArguments(args, benchmark, views,
#     #                                               views_indices,
#     #                                               argument_dictionaries,
#     #                                               random_state, directory,
#     #                                               resultsMonoview,
#     #                                               classification_indices)
#     logging.debug("Done:\t multiview arguments initialization")
#
#     logging.debug("Start:\t multiview benchmark")
#     results_multiview = []
#     nb_experiments = len(argument_dictionaries["multiview"])
#     nb_multicore_to_do = int(math.ceil(float(nb_experiments) / nb_cores))
#     for step_index in range(nb_multicore_to_do):
#         results_multiview += Parallel(n_jobs=nb_cores)(
#             delayed(exec_multiview_multicore)(directory, core_index, args["name"],
#                                               classification_indices, k_folds,
#                                               args["file_type"], args["Base"]["pathf"],
#                                               labels_dictionary, random_state,
#                                               labels,
#                                               hyper_param_search=hyper_param_search,
#                                               metrics=metrics,
#                                               n_iter=args["hps_iter"],
#                                               **
#                                              argument_dictionaries["multiview"][
#                                                  step_index * nb_cores + core_index])
#             for core_index in
#             range(min(nb_cores, nb_experiments - step_index * nb_cores)))
#     logging.debug("Done:\t multiview benchmark")
#
#     return [flag, results_monoview + results_multiview]


def exec_one_benchmark_mono_core(dataset_var=None, labels_dictionary=None,
                                 directory=None, classification_indices=None,
                                 args=None,
                                 k_folds=None, random_state=None,
                                 hyper_param_search=None, metrics=None,
                                 argument_dictionaries=None,
                                 benchmark=None, views=None, views_indices=None,
                                 flag=None, labels=None, track_tracebacks=False):
    results_monoview, labels_names = benchmark_init(directory,
                                                 classification_indices, labels,
                                                 labels_dictionary, k_folds, dataset_var)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.debug("Start:\t monoview benchmark")
    traceback_outputs = {}
    for arguments in argument_dictionaries["monoview"]:
        try:
            X = dataset_var.get_v(arguments["view_index"])
            Y = dataset_var.get_labels()
            results_monoview += [
                exec_monoview(directory, X, Y, args["name"], labels_names,
                              classification_indices, k_folds,
                              1, args["file_type"], args["pathf"], random_state,
                              hyper_param_search=hyper_param_search, metrics=metrics,
                              n_iter=args["hps_iter"], **arguments)]
        except:
            if track_tracebacks:
                traceback_outputs[arguments["classifier_name"]+"-"+arguments["view_name"]] = traceback.format_exc()
            else:
                raise

    logging.debug("Done:\t monoview benchmark")

    logging.debug("Start:\t multiview arguments initialization")

    # argument_dictionaries = initMultiviewArguments(args, benchmark, views,
    #                                               views_indices,
    #                                               argument_dictionaries,
    #                                               random_state, directory,
    #                                               resultsMonoview,
    #                                               classification_indices)
    logging.debug("Done:\t multiview arguments initialization")

    logging.debug("Start:\t multiview benchmark")
    results_multiview = []
    for arguments in argument_dictionaries["multiview"]:
        try:
            results_multiview += [
                exec_multiview(directory, dataset_var, args["name"], classification_indices,
                              k_folds, 1, args["file_type"],
                              args["pathf"], labels_dictionary, random_state, labels,
                              hyper_param_search=hyper_param_search,
                              metrics=metrics, n_iter=args["hps_iter"], **arguments)]
        except:
            if track_tracebacks:
                traceback_outputs[arguments["classifier_name"]] = traceback.format_exc()
            else:
                raise
    logging.debug("Done:\t multiview benchmark")

    return [flag, results_monoview + results_multiview, traceback_outputs]


def exec_benchmark(nb_cores, stats_iter,
                   benchmark_arguments_dictionaries,
                   directory,  metrics, dataset_var, track_tracebacks,
                   exec_one_benchmark_mono_core=exec_one_benchmark_mono_core,
                   get_results=get_results, delete=delete_HDF5,
                   analyze_iterations=analyze_iterations):
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
    logging.debug("Start:\t Executing all the needed biclass benchmarks")
    results = []
    # if nb_cores > 1:
    #     if stats_iter > 1 or nb_multiclass > 1:
    #         nb_exps_to_do = len(benchmark_arguments_dictionaries)
    #         nb_multicore_to_do = range(int(math.ceil(float(nb_exps_to_do) / nb_cores)))
    #         for step_index in nb_multicore_to_do:
    #             results += (Parallel(n_jobs=nb_cores)(delayed(exec_one_benchmark)
    #                                                  (core_index=core_index,
    #                                                   **
    #                                                   benchmark_arguments_dictionaries[
    #                                                       core_index + step_index * nb_cores])
    #                                                  for core_index in range(
    #                 min(nb_cores, nb_exps_to_do - step_index * nb_cores))))
    #     else:
    #         results += [exec_one_benchmark_multicore(nb_cores=nb_cores, **
    #         benchmark_arguments_dictionaries[0])]
    # else:
    for arguments in benchmark_arguments_dictionaries:
        benchmark_results = exec_one_benchmark_mono_core(dataset_var=dataset_var,
                                                         track_tracebacks=track_tracebacks,
                                                         **arguments)
        analyze_iterations([benchmark_results], benchmark_arguments_dictionaries, stats_iter, metrics, example_ids=dataset_var.example_ids, labels=dataset_var.get_labels())
        results += [benchmark_results]
    logging.debug("Done:\t Executing all the needed biclass benchmarks")

    # Do everything with flagging
    logging.debug("Start:\t Analyzing predictions")
    results_mean_stds = get_results(results, stats_iter,
                                    benchmark_arguments_dictionaries,
                                    metrics,
                                    directory,
                                    dataset_var.example_ids,
                                    dataset_var.get_labels())
    logging.debug("Done:\t Analyzing predictions")
    delete(benchmark_arguments_dictionaries, nb_cores, dataset_var)
    return results_mean_stds


def exec_classif(arguments):
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
    os.nice(args["nice"])
    nb_cores = args["nb_cores"]
    if nb_cores == 1:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
    stats_iter = args["stats_iter"]
    hyper_param_search = args["hps_type"]
    cl_type = args["type"]
    monoview_algos = args["algos_monoview"]
    multiview_algos = args["algos_multiview"]
    dataset_list = execution.find_dataset_names(args["pathf"],
                                                args["file_type"],
                                                args["name"])
    if not args["add_noise"]:
        args["noise_std"]=[0.0]
    for dataset_name in dataset_list:
        noise_results = []
        for noise_std in args["noise_std"]:

            directory = execution.init_log_file(dataset_name, args["views"], args["file_type"],
                                              args["log"], args["debug"], args["label"],
                                              args["res_dir"], args["add_noise"], noise_std, args)

            random_state = execution.init_random_state(args["random_state"], directory)
            stats_iter_random_states = execution.init_stats_iter_random_states(stats_iter,
                                                                        random_state)

            get_database = execution.get_database_function(dataset_name, args["file_type"])

            dataset_var, labels_dictionary, datasetname = get_database(args["views"],
                                                                  args["pathf"], dataset_name,
                                                                  args["nb_class"],
                                                                  args["classes"],
                                                                  random_state,
                                                                  args["full"],
                                                                  args["add_noise"],
                                                                  noise_std)
            args["name"] = datasetname

            splits = execution.gen_splits(dataset_var.get_labels(), args["split"],
                                         stats_iter_random_states)

            # multiclass_labels, labels_combinations, indices_multiclass = multiclass.gen_multiclass_labels(
            #     dataset_var.get_labels(), multiclass_method, splits)

            k_folds = execution.gen_k_folds(stats_iter, args["nb_folds"],
                                         stats_iter_random_states)

            dataset_files = dataset.init_multiple_datasets(args["pathf"], args["name"], nb_cores)


            views, views_indices, all_views = execution.init_views(dataset_var, args["views"])
            views_dictionary = dataset_var.get_view_dict()
            nb_views = len(views)
            nb_class = dataset_var.get_nb_class()

            metrics = [metric.split(":") for metric in args["metrics"]]
            if metrics == [["all"]]:
                metrics_names = [name for _, name, isPackage
                                in pkgutil.iter_modules(
                        [os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'metrics')]) if
                                not isPackage and name not in ["framework", "log_loss",
                                                               "matthews_corrcoef",
                                                               "roc_auc_score"]]
                metrics = [[metricName] for metricName in metrics_names]
            metrics = arange_metrics(metrics, args["metric_princ"])
            for metricIndex, metric in enumerate(metrics):
                if len(metric) == 1:
                    metrics[metricIndex] = [metric[0], None]

            benchmark = init_benchmark(cl_type, monoview_algos, multiview_algos, args)
            init_kwargs= init_kwargs_func(args, benchmark)
            data_base_time = time.time() - start
            argument_dictionaries = init_argument_dictionaries(
                benchmark, views_dictionary,
                nb_class, init_kwargs)
            # argument_dictionaries = initMonoviewExps(benchmark, viewsDictionary,
            #                                         NB_CLASS, initKWARGS)
            directories = execution.gen_direcorties_names(directory, stats_iter)
            benchmark_argument_dictionaries = execution.gen_argument_dictionaries(
                labels_dictionary, directories,
                splits,
                hyper_param_search, args, k_folds,
                stats_iter_random_states, metrics,
                argument_dictionaries, benchmark,
                views, views_indices,)
            results_mean_stds = exec_benchmark(
                nb_cores, stats_iter,
                benchmark_argument_dictionaries, directory, metrics, dataset_var,
                args["track_tracebacks"])
            noise_results.append([noise_std, results_mean_stds])
            plot_results_noise(directory, noise_results, metrics[0][0], dataset_name)


