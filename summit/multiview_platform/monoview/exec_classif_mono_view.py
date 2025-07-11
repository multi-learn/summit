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
#!/usr/bin/env python

""" Execution: Script to perform a MonoView classification """

import logging  # To create Log-Files
# Import built-in modules
import os  # to geth path of the running script
import time  # for time calculations

import h5py
# Import 3rd party modules
import numpy as np  # for reading CSV-files and Series

from .monoview_utils import MonoviewResult, MonoviewResultAnalyzer
# Import own modules
from .. import monoview_classifiers
from ..utils import hyper_parameter_search
from ..utils.dataset import extract_subset, HDF5Dataset
from ..utils.multiclass import get_mc_estim
from ..utils.organization import secure_file_path

# Author-Info
__author__ = "Baptiste BAUVIN"
__status__ = "Prototype"  # Production, Development, Prototype


# __date__ = 2016 - 03 - 25


def exec_monoview_multicore(directory, name, labels_names,
                            classification_indices,
                            k_folds, dataset_file_index, database_type,
                            path, random_state, labels,
                            hyper_param_search="randomized_search",
                            metrics=[["accuracy_score", None]], n_iter=30,
                            **args):  # pragma: no cover
    dataset_var = HDF5Dataset(
        hdf5_file=h5py.File(path + name + str(dataset_file_index) + ".hdf5", "r"))
    needed_view_index = args["view_index"]
    X = dataset_var.get_v(needed_view_index)
    Y = labels
    return exec_monoview(directory, X, Y, name, labels_names,
                         classification_indices, k_folds, 1, database_type,
                         path,
                         random_state, hyper_param_search=hyper_param_search,
                         metrics=metrics, n_iter=n_iter,
                         view_name=dataset_var.get_view_name(
                             args["view_index"]),
                         **args)


def exec_monoview(directory, X, Y, database_name, labels_names,
                  classification_indices,
                  k_folds, nb_cores, databaseType, path,
                  random_state, hyper_param_search="Random",
                  metrics={"accuracy_score*": {}}, n_iter=30, view_name="",
                  hps_kwargs={}, feature_ids=[], **args):
    logging.info("Start:\t Loading data")
    kwargs, \
        t_start, \
        view_name, \
        classifier_name, \
        X, \
        learning_rate, \
        labels_string, \
        output_file_name, \
        directory, \
        base_file_name = init_constants(args, X, classification_indices,
                                        labels_names,
                                        database_name, directory, view_name, )
    logging.info("Done:\t Loading data")

    logging.info(
        "Info:\t Classification - Database:" + str(
            database_name) + " View:" + str(
            view_name) + " train ratio:"
        + str(learning_rate) + ", CrossValidation k-folds: " + str(
            k_folds.n_splits) + ", cores:"
        + str(nb_cores) + ", algorithm : " + classifier_name)

    logging.info("Start:\t Determine Train/Test split")
    X_train, y_train, X_test, y_test = init_train_test(X, Y,
                                                       classification_indices)

    logging.info("Info:\t Shape X_train:" + str(
        X_train.shape) + ", Length of y_train:" + str(len(y_train)))
    logging.info("Info:\t Shape X_test:" + str(
        X_test.shape) + ", Length of y_test:" + str(len(y_test)))
    logging.info("Done:\t Determine Train/Test split")

    logging.info("Start:\t Generate classifier args")
    classifier_module = getattr(monoview_classifiers, classifier_name)
    classifier_class_name = classifier_module.classifier_class_name
    hyper_param_beg = time.monotonic()
    cl_kwargs = get_hyper_params(classifier_module, hyper_param_search,
                                 classifier_name,
                                 classifier_class_name,
                                 X_train, y_train,
                                 random_state, output_file_name,
                                 k_folds, nb_cores, metrics, kwargs,
                                 **hps_kwargs)
    hyper_param_duration = time.monotonic() - hyper_param_beg
    logging.info("Done:\t Generate classifier args")

    logging.info("Start:\t Training")

    classifier = get_mc_estim(getattr(classifier_module,
                                      classifier_class_name)
                              (random_state, **cl_kwargs),
                              random_state,
                              y=Y)
    fit_beg = time.monotonic()
    classifier.fit(X_train, y_train)
    fit_duration = time.monotonic() - fit_beg
    logging.info("Done:\t Training")

    logging.info("Start:\t Predicting")
    train_pred = classifier.predict(X_train)
    pred_beg = time.monotonic()
    test_pred = classifier.predict(X_test)
    pred_duration = time.monotonic() - pred_beg

    # Filling the full prediction in the right order
    full_pred = np.zeros(Y.shape, dtype=int) - 100
    for train_index, index in enumerate(classification_indices[0]):
        full_pred[index] = train_pred[train_index]
    for test_index, index in enumerate(classification_indices[1]):
        full_pred[index] = test_pred[test_index]

    logging.info("Done:\t Predicting")

    whole_duration = time.monotonic() - t_start
    logging.info(
        "Info:\t Duration for training and predicting: " + str(
            whole_duration) + "[s]")

    logging.info("Start:\t Getting results")
    result_analyzer = MonoviewResultAnalyzer(view_name=view_name,
                                             classifier_name=classifier_name,
                                             shape=X.shape,
                                             classifier=classifier,
                                             classification_indices=classification_indices,
                                             k_folds=k_folds,
                                             hps_method=hyper_param_search,
                                             metrics_dict=metrics,
                                             n_iter=n_iter,
                                             class_label_names=labels_names,
                                             pred=full_pred,
                                             directory=directory,
                                             base_file_name=base_file_name,
                                             labels=Y,
                                             database_name=database_name,
                                             nb_cores=nb_cores,
                                             duration=whole_duration,
                                             feature_ids=feature_ids)
    string_analysis, images_analysis, metrics_scores, class_metrics_scores, \
        confusion_matrix = result_analyzer.analyze()
    logging.info("Done:\t Getting results")

    logging.info("Start:\t Saving preds")
    save_results(string_analysis, output_file_name, full_pred, train_pred,
                 y_train, images_analysis, y_test, confusion_matrix)
    logging.info("Done:\t Saving results")

    view_index = args["view_index"]
    return MonoviewResult(view_index, classifier_name, view_name,
                          metrics_scores, full_pred, cl_kwargs,
                          classifier, X_train.shape[1],
                          hyper_param_duration, fit_duration, pred_duration,
                          class_metrics_scores)


def init_constants(args, X, classification_indices, labels_names,
                   name, directory, view_name):
    try:
        kwargs = args["args"]
    except KeyError:
        kwargs = args
    t_start = time.monotonic()
    cl_type = kwargs["classifier_name"]
    learning_rate = float(len(classification_indices[0])) / (
        len(classification_indices[0]) + len(classification_indices[1]))
    labels_string = "-".join(labels_names)
    cl_type_string = cl_type
    directory = os.path.join(directory, cl_type_string, view_name, )
    base_file_name = cl_type_string + '-' + name + "-" + view_name + "-"
    output_file_name = os.path.join(directory, base_file_name)
    secure_file_path(output_file_name)
    return kwargs, t_start, view_name, cl_type, X, learning_rate, labels_string,\
           output_file_name, directory, base_file_name


def init_train_test(X, Y, classification_indices):
    train_indices, test_indices = classification_indices
    X_train = extract_subset(X, train_indices)
    X_test = extract_subset(X, test_indices)
    y_train = Y[train_indices]
    y_test = Y[test_indices]
    return X_train, y_train, X_test, y_test


def get_hyper_params(classifier_module, search_method, classifier_module_name,
                     classifier_class_name, X_train, y_train,
                     random_state,
                     output_file_name, k_folds, nb_cores, metrics, kwargs,
                     **hps_kwargs):
    if search_method != "None":
        logging.info(
            "Start:\t " + search_method + " best settings for " + classifier_module_name)
        classifier_hp_search = getattr(hyper_parameter_search, search_method)
        estimator = getattr(classifier_module, classifier_class_name)(
            random_state=random_state,
            **kwargs[classifier_module_name])
        estimator = get_mc_estim(estimator, random_state,
                                 multiview=False, y=y_train)
        hps = classifier_hp_search(estimator, scoring=metrics, cv=k_folds,
                                   random_state=random_state,
                                   framework="monoview", n_jobs=nb_cores,
                                   **hps_kwargs)
        hps.fit(X_train, y_train)
        cl_kwargs = hps.get_best_params()
        hps.gen_report(output_file_name)
        logging.info("Done:\t " + search_method + " best settings")
    else:
        cl_kwargs = kwargs[classifier_module_name]
    return cl_kwargs


def save_results(string_analysis, output_file_name, full_labels_pred,
                 y_train_pred,
                 y_train, images_analysis, y_test,
                 confusion_matrix):  # pragma: no cover
    logging.info(string_analysis)
    output_text_file = open(output_file_name + 'summary.txt', 'w',
                            encoding="utf-8")
    output_text_file.write(string_analysis)
    output_text_file.close()
    np.savetxt(output_file_name + "confusion_matrix.csv", confusion_matrix,
               delimiter=', ')
    np.savetxt(output_file_name + "full_pred.csv",
               full_labels_pred.astype(np.int16), delimiter=",")
    np.savetxt(output_file_name + "train_pred.csv",
               y_train_pred.astype(np.int16),
               delimiter=",")
    np.savetxt(output_file_name + "train_labels.csv", y_train.astype(np.int16),
               delimiter=",")
    np.savetxt(output_file_name + "test_labels.csv", y_test.astype(np.int16),
               delimiter=",")

    if images_analysis is not None:
        for image_name in images_analysis:
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
