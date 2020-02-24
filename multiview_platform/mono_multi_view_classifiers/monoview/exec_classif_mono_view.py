#!/usr/bin/env python

""" Execution: Script to perform a MonoView classification """

import errno
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

# Author-Info
__author__ = "Nikolas Huelsmann, Baptiste BAUVIN"
__status__ = "Prototype"  # Production, Development, Prototype


# __date__ = 2016 - 03 - 25


def exec_monoview_multicore(directory, name, labels_names,
                            classification_indices,
                            k_folds, dataset_file_index, database_type,
                            path, random_state, labels,
                            hyper_param_search="randomized_search",
                            metrics=[["accuracy_score", None]], n_iter=30,
                            **args):
    dataset_var = HDF5Dataset(
        hdf5_file=h5py.File(path + name + str(dataset_file_index) + ".hdf5",
                            "r"))
    neededViewIndex = args["view_index"]
    X = dataset_var.get_v(neededViewIndex)
    Y = labels
    return exec_monoview(directory, X, Y, name, labels_names,
                         classification_indices, k_folds, 1, database_type,
                         path,
                         random_state, hyper_param_search=hyper_param_search,
                         metrics=metrics, n_iter=n_iter,
                         view_name=dataset_var.get_view_name(
                             args["view_index"]),
                         **args)


def exec_monoview(directory, X, Y, database_name, labels_names, classification_indices,
                  k_folds, nb_cores, databaseType, path,
                  random_state, hyper_param_search="randomized_search",
                  metrics=[["accuracy_score", None]], n_iter=30, view_name="",
                  **args):
    logging.debug("Start:\t Loading data")
    kwargs, \
    t_start, \
    view_name, \
    classifier_name, \
    X, \
    learningRate, \
    labelsString, \
    output_file_name = init_constants(args, X, classification_indices,
                                      labels_names,
                                      database_name, directory, view_name)
    logging.debug("Done:\t Loading data")

    logging.debug(
        "Info:\t Classification - Database:" + str(database_name) + " View:" + str(
            view_name) + " train ratio:"
        + str(learningRate) + ", CrossValidation k-folds: " + str(
            k_folds.n_splits) + ", cores:"
        + str(nb_cores) + ", algorithm : " + classifier_name)

    logging.debug("Start:\t Determine Train/Test split")
    X_train, y_train, X_test, y_test = init_train_test(X, Y,
                                                       classification_indices)

    logging.debug("Info:\t Shape X_train:" + str(
        X_train.shape) + ", Length of y_train:" + str(len(y_train)))
    logging.debug("Info:\t Shape X_test:" + str(
        X_test.shape) + ", Length of y_test:" + str(len(y_test)))
    logging.debug("Done:\t Determine Train/Test split")

    logging.debug("Start:\t Generate classifier args")
    classifier_module = getattr(monoview_classifiers, classifier_name)
    classifier_class_name = classifier_module.classifier_class_name
    cl_kwargs, test_folds_preds = get_hyper_params(classifier_module, hyper_param_search,
                                                   n_iter, classifier_name,
                                                   classifier_class_name,
                                                   X_train, y_train,
                                                   random_state, output_file_name,
                                                   k_folds, nb_cores, metrics, kwargs)
    logging.debug("Done:\t Generate classifier args")

    logging.debug("Start:\t Training")

    classifier = get_mc_estim(getattr(classifier_module,
                                      classifier_class_name)
                              (random_state, **cl_kwargs),
                              random_state,
                              y=Y)

    classifier.fit(X_train, y_train)  # NB_CORES=nbCores,
    logging.debug("Done:\t Training")

    logging.debug("Start:\t Predicting")
    train_pred = classifier.predict(X_train)
    test_pred = classifier.predict(X_test)

    # Filling the full prediction in the right order
    full_pred = np.zeros(Y.shape, dtype=int) - 100
    for trainIndex, index in enumerate(classification_indices[0]):
        full_pred[index] = train_pred[trainIndex]
    for testIndex, index in enumerate(classification_indices[1]):
        full_pred[index] = test_pred[testIndex]

    logging.debug("Done:\t Predicting")

    duration = time.time() - t_start
    logging.debug(
        "Info:\t Time for training and predicting: " + str(duration) + "[s]")

    logging.debug("Start:\t Getting results")
    result_analyzer = MonoviewResultAnalyzer(view_name=view_name,
                                             classifier_name=classifier_name,
                                             shape=X.shape,
                                             classifier=classifier,
                                             classification_indices=classification_indices,
                                             k_folds=k_folds,
                                             hps_method=hyper_param_search,
                                             metrics_list=metrics,
                                             n_iter=n_iter,
                                             class_label_names=labels_names,
                                             train_pred=train_pred,
                                             test_pred=test_pred,
                                             directory=output_file_name,
                                             labels=Y,
                                             database_name=database_name,
                                             nb_cores=nb_cores,
                                             duration=duration)
    string_analysis, images_analysis, metrics_scores = result_analyzer.analyze()
    logging.debug("Done:\t Getting results")

    logging.debug("Start:\t Saving preds")
    save_results(string_analysis, output_file_name, full_pred, train_pred,
                 y_train, images_analysis, y_test)
    logging.info("Done:\t Saving results")

    view_index = args["view_index"]
    if test_folds_preds is None:
        test_folds_preds = train_pred
    return MonoviewResult(view_index, classifier_name, view_name,
                                         metrics_scores,
                                         full_pred, cl_kwargs,
                                         test_folds_preds, classifier,
                                         X_train.shape[1])


def init_constants(args, X, classification_indices, labels_names,
                   name, directory, view_name):
    try:
        kwargs = args["args"]
    except KeyError:
        kwargs = args
    t_start = time.time()
    cl_type = kwargs["classifier_name"]
    learning_rate = float(len(classification_indices[0])) / (
            len(classification_indices[0]) + len(classification_indices[1]))
    labels_string = "-".join(labels_names)
    cl_type_string = cl_type
    output_file_name = os.path.join(directory, cl_type_string, view_name,
                                    cl_type_string + '-' + name + "-" +
                                    view_name + "-")
    if not os.path.exists(os.path.dirname(output_file_name)):
        try:
            os.makedirs(os.path.dirname(output_file_name))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return kwargs, t_start, view_name, cl_type, X, learning_rate, labels_string, output_file_name


def init_train_test(X, Y, classification_indices):
    train_indices, test_indices = classification_indices
    X_train = extract_subset(X, train_indices)
    X_test = extract_subset(X, test_indices)
    y_train = Y[train_indices]
    y_test = Y[test_indices]
    return X_train, y_train, X_test, y_test


def get_hyper_params(classifier_module, hyper_param_search, nIter, classifier_module_name,
                     classifier_class_name, X_train, y_train,
                     random_state,
                     output_file_name, k_folds, nb_cores, metrics, kwargs):
    if hyper_param_search != "None":
        logging.debug(
            "Start:\t " + hyper_param_search + " best settings with " + str(
                nIter) + " iterations for " + classifier_module_name)
        classifier_hp_search = getattr(hyper_parameter_search,
                                       hyper_param_search.split("-")[0])
        cl_kwargs, test_folds_preds = classifier_hp_search(X_train, y_train,
                                                           "monoview",
                                                           random_state,
                                                           output_file_name,
                                                           classifier_module,
                                                           classifier_class_name,
                                                           folds=k_folds,
                                                           nb_cores=nb_cores,
                                                           metric=metrics[0],
                                                           n_iter=nIter,
                                                           classifier_kwargs=
                                                           kwargs[
                                                               classifier_module_name])
        logging.debug("Done:\t " + hyper_param_search + " best settings")
    else:
        cl_kwargs = kwargs[classifier_module_name]
        test_folds_preds = None
    return cl_kwargs, test_folds_preds


def save_results(string_analysis, output_file_name, full_labels_pred,
                 y_train_pred,
                 y_train, images_analysis, y_test):
    logging.info(string_analysis)
    output_text_file = open(output_file_name + 'summary.txt', 'w')
    output_text_file.write(string_analysis)
    output_text_file.close()
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
