import logging
import os
import os.path
import time

import h5py
import numpy as np

from .multiview_utils import MultiviewResult, MultiviewResultAnalyzer
from .. import multiview_classifiers
from ..utils import hyper_parameter_search
from ..utils.multiclass import get_mc_estim
from ..utils.organization import secure_file_path

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def init_constants(kwargs, classification_indices, metrics,
                   name, nb_cores, k_folds,
                   dataset_var, directory):
    """
    Used to init the constants
    Parameters
    ----------
    kwargs :

    classification_indices :

    metrics :

    name :

    nb_cores : nint number of cares to execute

    k_folds :

    dataset_var :  {array-like} shape (n_samples, n_features)
        dataset variable

    Returns
    -------
    tuple of (classifier_name, t_start, views_indices,
              classifier_config, views, learning_rate)
    """
    views = kwargs["view_names"]
    views_indices = kwargs["view_indices"]
    if not metrics:
        metrics = [["f1_score", None]]
    classifier_name = kwargs["classifier_name"]
    classifier_config = kwargs[classifier_name]
    learning_rate = len(classification_indices[0]) / float(
        (len(classification_indices[0]) + len(classification_indices[1])))
    t_start = time.time()
    logging.info("Info\t: Classification - Database : " + str(
        name) + " ; Views : " + ", ".join(views) +
                 " ; Algorithm : " + classifier_name + " ; Cores : " + str(
        nb_cores) + ", Train ratio : " + str(learning_rate) +
                 ", CV on " + str(k_folds.n_splits) + " folds")

    for view_index, view_name in zip(views_indices, views):
        logging.info("Info:\t Shape of " + str(view_name) + " :" + str(
            dataset_var.get_shape()))
    labels = dataset_var.get_labels()
    directory = os.path.join(directory, classifier_name)
    base_file_name = classifier_name+"-"+dataset_var.get_name()+"-"
    output_file_name = os.path.join(directory, base_file_name)
    return classifier_name, t_start, views_indices, \
           classifier_config, views, learning_rate, labels, output_file_name,\
           directory, base_file_name


def save_results(string_analysis, images_analysis, output_file_name):
    """
    Save results in derectory

    Parameters
    ----------

    classifier : classifier class

    labels_dictionary : dict dictionary of labels

    string_analysis : str

    views :

    classifier_module : module of the classifier

    classification_kargs :

    directory : str directory

    learning_rate :

    name :

    images_analysis :

    """
    logging.info(string_analysis)
    secure_file_path(output_file_name)
    output_text_file = open(output_file_name + 'summary.txt', 'w')
    output_text_file.write(string_analysis)
    output_text_file.close()

    if images_analysis is not None:
        for image_name in images_analysis.keys():
            if os.path.isfile(output_file_name + image_name + ".png"):
                for i in range(1, 20):
                    test_file_name = output_file_name + image_name + "-" + str(
                        i) + ".png"
                    if not os.path.isfile(test_file_name):
                        images_analysis[image_name].savefig(test_file_name,
                                                            transparent=True)
                        break

            images_analysis[image_name].savefig(
                output_file_name + image_name + '.png', transparent=True)


def exec_multiview_multicore(directory, core_index, name, learning_rate,
                             nb_folds,
                             database_type, path, labels_dictionary,
                             random_state, labels,
                             hyper_param_search=False, nb_cores=1, metrics=None,
                             n_iter=30, **arguments):
    """
    execute multiview process on

    Parameters
    ----------

    directory : indicate the directory

    core_index :

    name : name of the data file to perform

    learning_rate :

    nb_folds :

    database_type :

    path : path to the data name

    labels_dictionary

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    labels :

    hyper_param_search :

    nb_cores : in number of cores

    metrics : metric to use

    n_iter : int number of iterations

    arguments : others arguments

    Returns
    -------
    exec_multiview on directory, dataset_var, name, learning_rate, nb_folds, 1,
        database_type, path, labels_dictionary,
        random_state, labels,
        hyper_param_search=hyper_param_search, metrics=metrics,
        n_iter=n_iter, **arguments
    """
    """Used to load an HDF5 dataset_var for each parallel job and execute multiview classification"""
    dataset_var = h5py.File(path + name + str(core_index) + ".hdf5", "r")
    return exec_multiview(directory, dataset_var, name, learning_rate, nb_folds,
                          1,
                          database_type, path, labels_dictionary,
                          random_state, labels,
                          hps_method=hyper_param_search,
                          metrics=metrics,
                          n_iter=n_iter, **arguments)


def exec_multiview(directory, dataset_var, name, classification_indices,
                   k_folds,
                   nb_cores, database_type, path,
                   labels_dictionary, random_state, labels,
                   hps_method="None", hps_kwargs={}, metrics=None,
                   n_iter=30, **kwargs):
    """Used to execute multiview classification and result analysis

    Parameters
    ----------

    directory : indicate the directory


    dataset_var :

    name

    classification_indices

    k_folds

    nb_cores

    database_type

    path

    labels_dictionary : dict dictionary of labels

    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    labels

    hps_method

    metrics

    n_iter : int number of iterations

    kwargs

    Returns
    -------

    ``MultiviewResult``
    """

    logging.debug("Start:\t Initialize constants")
    cl_type, \
    t_start, \
    views_indices, \
    classifier_config, \
    views, \
    learning_rate, \
    labels, \
    output_file_name,\
    directory,\
    base_file_name = init_constants(kwargs, classification_indices, metrics, name,
                            nb_cores, k_folds, dataset_var, directory)
    logging.debug("Done:\t Initialize constants")

    extraction_time = time.time() - t_start
    logging.info("Info:\t Extraction duration " + str(extraction_time) + "s")

    logging.debug("Start:\t Getting train/test split")
    learning_indices, validation_indices = classification_indices
    logging.debug("Done:\t Getting train/test split")

    logging.debug("Start:\t Getting classifiers modules")
    # classifierPackage = getattr(multiview_classifiers,
    #                             CL_type)  # Permet d'appeler un module avec une string
    classifier_module = getattr(multiview_classifiers, cl_type)
    classifier_name = classifier_module.classifier_class_name
    # classifierClass = getattr(classifierModule, CL_type + "Class")
    logging.debug("Done:\t Getting classifiers modules")

    logging.debug("Start:\t Optimizing hyperparameters")
    hps_beg = time.monotonic()
    if hps_method != "None":
        hps_method_class = getattr(hyper_parameter_search, hps_method)
        estimator = getattr(classifier_module, classifier_name)(
                    random_state=random_state,
                    **classifier_config)
        estimator = get_mc_estim(estimator, random_state,
                                         multiview=True,
                                         y=dataset_var.get_labels()[learning_indices])
        hps = hps_method_class(estimator, scoring=metrics, cv=k_folds,
                               random_state=random_state, framework="multiview",
                               n_jobs=nb_cores,
                               learning_indices=learning_indices,
                               view_indices=views_indices, **hps_kwargs)
        hps.fit(dataset_var, dataset_var.get_labels(), )
        classifier_config = hps.get_best_params()
        hps.gen_report(output_file_name)
        # classifier_config = hyper_parameter_search.search_best_settings(
        #     dataset_var, dataset_var.get_labels(), classifier_module,
        #     classifier_name,
        #     metrics[0], learning_indices, k_folds, random_state,
        #     output_file_name, nb_cores=nb_cores, views_indices=views_indices,
        #     searching_tool=hps_method, n_iter=n_iter,
        #     classifier_config=classifier_config)
    hps_duration = time.monotonic() - hps_beg
    classifier = get_mc_estim(
        getattr(classifier_module, classifier_name)(random_state=random_state,
                                                    **classifier_config),
        random_state, multiview=True,
        y=dataset_var.get_labels())
    logging.debug("Done:\t Optimizing hyperparameters")
    logging.debug("Start:\t Fitting classifier")
    fit_beg = time.monotonic()
    classifier.fit(dataset_var, dataset_var.get_labels(),
                   train_indices=learning_indices,
                   view_indices=views_indices)
    fit_duration = time.monotonic() - fit_beg
    logging.debug("Done:\t Fitting classifier")

    logging.debug("Start:\t Predicting")
    train_pred = classifier.predict(dataset_var,
                                           example_indices=learning_indices,
                                           view_indices=views_indices)
    pred_beg = time.monotonic()
    test_pred = classifier.predict(dataset_var,
                                          example_indices=validation_indices,
                                          view_indices=views_indices)
    pred_duration = time.monotonic() - pred_beg
    full_labels = np.zeros(dataset_var.get_labels().shape, dtype=int) - 100
    full_labels[learning_indices] = train_pred
    full_labels[validation_indices] = test_pred
    logging.info("Done:\t Pertidcting")

    whole_duration = time.time() - t_start
    logging.info(
        "Info:\t Classification duration " + str(extraction_time) + "s")

    # TODO: get better cltype

    logging.info("Start:\t Result Analysis for " + cl_type)
    times = (extraction_time, whole_duration)
    result_analyzer = MultiviewResultAnalyzer(view_names=views,
                                              classifier=classifier,
                                              classification_indices=classification_indices,
                                              k_folds=k_folds,
                                              hps_method=hps_method,
                                              metrics_list=metrics,
                                              n_iter=n_iter,
                                              class_label_names=list(labels_dictionary.values()),
                                              train_pred=train_pred,
                                              test_pred=test_pred,
                                              directory=directory,
                                              base_file_name=base_file_name,
                                              labels=labels,
                                              database_name=dataset_var.get_name(),
                                              nb_cores=nb_cores,
                                              duration=whole_duration)
    string_analysis, images_analysis, metrics_scores = result_analyzer.analyze()
    logging.info("Done:\t Result Analysis for " + cl_type)

    logging.debug("Start:\t Saving preds")
    save_results(string_analysis, images_analysis, output_file_name)
    logging.debug("Start:\t Saving preds")

    return MultiviewResult(cl_type, classifier_config, metrics_scores,
                           full_labels, hps_duration, fit_duration,
                           pred_duration)
