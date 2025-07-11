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
import os

import h5py
import numpy as np

from .dataset import RAMDataset, HDF5Dataset
from .organization import secure_file_path

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def make_me_noisy(view_data, random_state, percentage=5):
    """used to introduce some noise in the generated data"""
    view_data = view_data.astype(bool)
    nb_noisy_coord = int(
        percentage / 100.0 * view_data.shape[0] * view_data.shape[1])
    rows = range(view_data.shape[0])
    cols = range(view_data.shape[1])
    for _ in range(nb_noisy_coord):
        row_idx = random_state.choice(rows)
        col_idx = random_state.choice(cols)
        view_data[row_idx, col_idx] = 0
    noisy_view_data = view_data.astype(np.uint8)
    return noisy_view_data


def get_plausible_db_hdf5(features, path, file_name, nb_class=3,
                          label_names=["No".encode(), "Yes".encode(),
                                       "Maybe".encode()],
                          random_state=None, full=True, add_noise=False,
                          noise_std=0.15, nb_view=3, nb_samples=100,
                          nb_features=10):
    """Used to generate a plausible dataset to test the algorithms"""
    secure_file_path(os.path.join(path, "plausible.hdf5"))
    sample_ids = ["exmaple_id_" + str(i) for i in range(nb_samples)]
    views = []
    view_names = []
    are_sparse = []
    if nb_class == 2:
        labels = np.array(
            [0 for _ in range(int(nb_samples / 2))] + [1 for _ in range(
                nb_samples - int(nb_samples / 2))])
        label_names = ["No".encode(), "Yes".encode()]
        for view_index in range(nb_view):
            view_data = np.array(
                [np.zeros(nb_features) for _ in range(int(nb_samples / 2))] +
                [np.ones(nb_features) for _ in
                 range(nb_samples - int(nb_samples / 2))])
            fake_one_indices = random_state.randint(0, int(nb_samples / 2),
                                                    int(nb_samples / 12))
            fake_zero_indices = random_state.randint(int(nb_samples / 2),
                                                     nb_samples,
                                                     int(nb_samples / 12))
            for index in np.concatenate((fake_one_indices, fake_zero_indices)):
                sample_ids[index] += "noised"

            view_data[fake_one_indices] = np.ones(
                (len(fake_one_indices), nb_features))
            view_data[fake_zero_indices] = np.zeros(
                (len(fake_zero_indices), nb_features))
            view_data = make_me_noisy(view_data, random_state)
            views.append(view_data)
            view_names.append("ViewNumber" + str(view_index))
            are_sparse.append(False)

        dataset = RAMDataset(views=views, labels=labels,
                             labels_names=label_names, view_names=view_names,
                             are_sparse=are_sparse, sample_ids=sample_ids,
                             name='plausible')
        labels_dictionary = {0: "No", 1: "Yes"}
        return dataset, labels_dictionary, "plausible"
    elif nb_class >= 3:
        firstBound = int(nb_samples / 3)
        rest = nb_samples - 2 * int(nb_samples / 3)
        scndBound = 2 * int(nb_samples / 3)
        thrdBound = nb_samples
        labels = np.array(
            [0 for _ in range(firstBound)] +
            [1 for _ in range(firstBound)] +
            [2 for _ in range(rest)]
        )
        for view_index in range(nb_view):
            view_data = np.array(
                [np.zeros(nb_features) for _ in range(firstBound)] +
                [np.ones(nb_features) for _ in range(firstBound)] +
                [np.ones(nb_features) + 1 for _ in range(rest)])
            fake_one_indices = random_state.randint(0, firstBound,
                                                    int(nb_samples / 12))
            fakeTwoIndices = random_state.randint(firstBound, scndBound,
                                                  int(nb_samples / 12))
            fake_zero_indices = random_state.randint(scndBound, thrdBound,
                                                     int(nb_samples / 12))

            view_data[fake_one_indices] = np.ones(
                (len(fake_one_indices), nb_features))
            view_data[fake_zero_indices] = np.zeros(
                (len(fake_zero_indices), nb_features))
            view_data[fakeTwoIndices] = np.ones(
                (len(fakeTwoIndices), nb_features)) + 1
            view_data = make_me_noisy(view_data, random_state)
            views.append(view_data)
            view_names.append("ViewNumber" + str(view_index))
            are_sparse.append(False)
        dataset = RAMDataset(views=views, labels=labels,
                             labels_names=label_names, view_names=view_names,
                             are_sparse=are_sparse,
                             name="plausible",
                             sample_ids=sample_ids)
        labels_dictionary = {0: "No", 1: "Yes", 2: "Maybe"}
        return dataset, labels_dictionary, "plausible"


class DatasetError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def get_classic_db_hdf5(views, path_f, name_DB, nb_class, asked_labels_names,
                        random_state, full=False, add_noise=False,
                        noise_std=0.15,
                        path_for_new="../data/"):
    """Used to load a hdf5 database"""
    if full:
        dataset_file = h5py.File(os.path.join(path_f, name_DB + ".hdf5"), "r")
        dataset = HDF5Dataset(hdf5_file=dataset_file)
        dataset_name = name_DB
        labels_dictionary = dict((label_index, label_name)
                                 for label_index, label_name
                                 in enumerate(dataset.get_label_names()))
    else:
        dataset_file = h5py.File(os.path.join(path_f, name_DB + ".hdf5"), "r")
        dataset = HDF5Dataset(hdf5_file=dataset_file)
        labels_dictionary = dataset.select_views_and_labels(nb_labels=nb_class,
                                                            selected_label_names=asked_labels_names,
                                                            view_names=views,
                                                            random_state=random_state,
                                                            path_for_new=path_for_new)
        dataset_name = dataset.get_name()

    if add_noise:
        dataset.add_gaussian_noise(random_state, path_for_new, noise_std)
        dataset_name = dataset.get_name()
    else:
        pass
    return dataset, labels_dictionary, dataset_name


def get_classic_db_csv(views, pathF, nameDB, NB_CLASS, askedLabelsNames,
                       random_state, full=False, add_noise=False,
                       noise_std=0.15,
                       delimiter=",", path_for_new="../data/"):
    # TODO : Update this one
    labels_names = np.genfromtxt(pathF + nameDB + "-labels-names.csv",
                                 dtype='str', delimiter=delimiter)
    datasetFile = h5py.File(pathF + nameDB + ".hdf5", "w")
    labels = np.genfromtxt(pathF + nameDB + "-labels.csv", delimiter=delimiter)
    labelsDset = datasetFile.create_dataset(
        "Labels", labels.shape, data=labels)
    labelsDset.attrs["names"] = [labelName.encode() for labelName in
                                 labels_names]
    viewFileNames = [viewFileName for viewFileName in
                     os.listdir(pathF + "Views/")]
    for viewIndex, viewFileName in enumerate(os.listdir(pathF + "Views/")):
        viewFile = pathF + "Views/" + viewFileName
        if viewFileName[-6:] != "-s.csv":
            viewMatrix = np.genfromtxt(viewFile, delimiter=delimiter)
            viewDset = datasetFile.create_dataset("View" + str(viewIndex),
                                                  viewMatrix.shape,
                                                  data=viewMatrix)
            del viewMatrix
            viewDset.attrs["name"] = viewFileName[:-4]
            viewDset.attrs["sparse"] = False
        else:
            pass
    metaDataGrp = datasetFile.create_group("Metadata")
    metaDataGrp.attrs["nbView"] = len(viewFileNames)
    metaDataGrp.attrs["nbClass"] = len(labels_names)
    metaDataGrp.attrs["datasetLength"] = len(labels)
    datasetFile.close()
    datasetFile, labelsDictionary, dataset_name = get_classic_db_hdf5(views,
                                                                      pathF,
                                                                      nameDB,
                                                                      NB_CLASS,
                                                                      askedLabelsNames,
                                                                      random_state,
                                                                      full,
                                                                      path_for_new=path_for_new)

    return datasetFile, labelsDictionary, dataset_name

#
# def get_classes(labels):
#     labels_set = set(list(labels))
#     nb_labels = len(labels_set)
#     if nb_labels >= 2:
#         return labels_set
#     else:
#         raise DatasetError("Dataset must have at least two different labels")
#
#
# def all_asked_labels_are_available(asked_labels_names_set,
#                                    available_labels_names):
#     for askedLabelName in asked_labels_names_set:
#         if askedLabelName in available_labels_names:
#             pass
#         else:
#             return False
#     return True
#
#
# def fill_label_names(nb_labels, selected_label_names, random_state,
#                      available_labels_names):
#     if len(selected_label_names) < nb_labels:
#         nb_labels_to_add = nb_labels - len(selected_label_names)
#         labels_names_to_choose = [available_label_name
#                                   for available_label_name
#                                   in available_labels_names
#                                   if available_label_name
#                                   not in selected_label_names]
#         added_labels_names = random_state.choice(labels_names_to_choose,
#                                               nb_labels_to_add, replace=False)
#         selected_label_names = list(selected_label_names) + list(added_labels_names)
#         asked_labels_names_set = set(selected_label_names)
#
#     elif len(selected_label_names) > nb_labels:
#         selected_label_names = list(
#             random_state.choice(selected_label_names, nb_labels, replace=False))
#         asked_labels_names_set = set(selected_label_names)
#
#     else:
#         asked_labels_names_set = set(selected_label_names)
#
#     return selected_label_names, asked_labels_names_set
#
#
# def get_all_labels(full_labels, available_labels_names):
#     new_labels = full_labels
#     new_labels_names = available_labels_names
#     used_indices = np.arange(len(full_labels))
#     return new_labels, new_labels_names, used_indices
#
#
# def select_asked_labels(asked_labels_names_set, available_labels_names,
#                         asked_labels_names, full_labels):
#     if all_asked_labels_are_available(asked_labels_names_set, available_labels_names):
#         used_labels = [available_labels_names.index(asked_label_name) for
#                       asked_label_name in asked_labels_names]
#         used_indices = np.array(
#             [labelIndex for labelIndex, label in enumerate(full_labels) if
#              label in used_labels])
#         new_labels = np.array([used_labels.index(label) for label in full_labels if
#                               label in used_labels])
#         new_labels_names = [available_labels_names[usedLabel] for usedLabel in
#                           used_labels]
#         return new_labels, new_labels_names, used_indices
#     else:
#         raise DatasetError("Asked labels are not all available in the dataset")
#
#
# def filter_labels(labels_set, asked_labels_names_set, full_labels,
#                   available_labels_names, asked_labels_names):
#     if len(labels_set) > 2:
#         if asked_labels_names == available_labels_names:
#             new_labels, new_labels_names, used_indices = \
#                 get_all_labels(full_labels, available_labels_names)
#         elif len(asked_labels_names_set) <= len(labels_set):
#             new_labels, new_labels_names, used_indices = select_asked_labels(
#                 asked_labels_names_set, available_labels_names,
#                 asked_labels_names, full_labels)
#         else:
#             raise DatasetError(
#                 "Asked more labels than available in the dataset. Available labels are : " +
#                 ", ".join(available_labels_names))
#
#     else:
#         new_labels, new_labels_names, used_indices = get_all_labels(full_labels,
#                                                                     available_labels_names)
#     return new_labels, new_labels_names, used_indices
#
#
# def filter_views(dataset_file, temp_dataset, views, used_indices):
#     new_view_index = 0
#     if views == [""]:
#         for view_index in range(dataset_file.get("Metadata").attrs["nbView"]):
#             copyhdf5_dataset(dataset_file, temp_dataset, "View" + str(view_index),
#                             "View" + str(view_index), used_indices)
#     else:
#         for asked_view_name in views:
#             for view_index in range(dataset_file.get("Metadata").attrs["nbView"]):
#                 view_name = dataset_file.get("View" + str(view_index)).attrs["name"]
#                 if type(view_name) == bytes:
#                     view_name = view_name.decode("utf-8")
#                 if view_name == asked_view_name:
#                     copyhdf5_dataset(dataset_file, temp_dataset,
#                                     "View" + str(view_index),
#                                     "View" + str(new_view_index), used_indices)
#                     new_view_name = \
#                     temp_dataset.get("View" + str(new_view_index)).attrs["name"]
#                     if type(new_view_name) == bytes:
#                         temp_dataset.get("View" + str(new_view_index)).attrs[
#                             "name"] = new_view_name.decode("utf-8")
#
#                     new_view_index += 1
#                 else:
#                     pass
#         temp_dataset.get("Metadata").attrs["nbView"] = len(views)
#
#
# def copyhdf5_dataset(source_data_file, destination_data_file, source_dataset_name,
#                      destination_dataset_name, used_indices):
#     """Used to copy a view in a new dataset file using only the samples of
#     usedIndices, and copying the args"""
#     new_d_set = destination_data_file.create_dataset(destination_dataset_name,
#                                                  data=source_data_file.get(
#                                                       source_dataset_name).value[
#                                                       used_indices, :])
#     if "sparse" in source_data_file.get(source_dataset_name).attrs.keys() and \
#             source_data_file.get(source_dataset_name).attrs["sparse"]:
#         # TODO : Support sparse
#         pass
#     else:
#         for key, value in source_data_file.get(source_dataset_name).attrs.items():
#             new_d_set.attrs[key] = value


#
# def add_gaussian_noise(dataset_file, random_state, path_f, dataset_name,
#                        noise_std=0.15):
#     """In this function, we add a guaussian noise centered in 0 with specified
#     std to each view, according to it's range (the noise will be
#     mutliplied by this range) and we crop the noisy signal according to the
#     view's attributes limits.
#     This is done by creating a new dataset, to keep clean data."""
#     noisy_dataset = h5py.File(path_f + dataset_name + "_noised.hdf5", "w")
#     dataset_file.copy("Metadata", noisy_dataset)
#     dataset_file.copy("Labels", noisy_dataset)
#     for view_index in range(dataset_file.get("Metadata").attrs["nbView"]):
#         dataset_file.copy("View" + str(view_index), noisy_dataset)
#     for view_index in range(noisy_dataset.get("Metadata").attrs["nbView"]):
#         view_name = "View" + str(view_index)
#         view_dset = noisy_dataset.get(view_name)
#         view_limits = dataset_file[
#             "Metadata/View" + str(view_index) + "_limits"].value
#         view_ranges = view_limits[:, 1] - view_limits[:, 0]
#         normal_dist = random_state.normal(0, noise_std, view_dset.value.shape)
#         noise = normal_dist * view_ranges
#         noised_data = view_dset.value + noise
#         noised_data = np.where(noised_data < view_limits[:, 0],
#                                view_limits[:, 0], noised_data)
#         noised_data = np.where(noised_data > view_limits[:, 1],
#                                view_limits[:, 1], noised_data)
#         noisy_dataset[view_name][...] = noised_data
#     original_dataset_filename = dataset_file.filename
#     dataset_file.close()
#     noisy_dataset.close()
#     noisy_dataset = h5py.File(path_f + dataset_name + "_noised.hdf5", "r")
#     if "_temp_" in original_dataset_filename:
#         os.remove(original_dataset_filename)
#     return noisy_dataset, dataset_name + "_noised"


# def getLabelSupports(CLASS_LABELS):
#     """Used to get the number of sample for each label"""
#     labels = set(CLASS_LABELS)
#     supports = [CLASS_LABELS.tolist().count(label) for label in labels]
# return supports, dict((label, index) for label, index in zip(labels,
# range(len(labels))))


# def isUseful(labelSupports, index, CLASS_LABELS, labelDict):
# if labelSupports[labelDict[CLASS_LABELS[index]]] != 0:
#     labelSupports[labelDict[CLASS_LABELS[index]]] -= 1
#     return True, labelSupports
# else:
#     return False, labelSupports


# def splitDataset(DATASET, LEARNING_RATE, DATASET_LENGTH, random_state):
#     LABELS = DATASET.get("Labels")[...]
#     NB_CLASS = int(DATASET["Metadata"].attrs["nbClass"])
#     validationIndices = extractRandomTrainingSet(LABELS, 1 - LEARNING_RATE, DATASET_LENGTH, NB_CLASS, random_state)
#     validationIndices.sort()
#     return validationIndices


# def extractRandomTrainingSet(CLASS_LABELS, LEARNING_RATE, DATASET_LENGTH, NB_CLASS, random_state):
#     labelSupports, labelDict = getLabelSupports(np.array(CLASS_LABELS))
#     nbTrainingExamples = [int(support * LEARNING_RATE) for support in labelSupports]
#     trainingExamplesIndices = []
#     usedIndices = []
#     while nbTrainingExamples != [0 for i in range(NB_CLASS)]:
#         isUseFull = False
#         index = int(random_state.randint(0, DATASET_LENGTH - 1))
#         if index not in usedIndices:
#             isUseFull, nbTrainingExamples = isUseful(nbTrainingExamples, index, CLASS_LABELS, labelDict)
#         if isUseFull:
#             trainingExamplesIndices.append(index)
#             usedIndices.append(index)
#     return trainingExamplesIndices


# def getKFoldIndices(nbFolds, CLASS_LABELS, NB_CLASS, learningIndices, random_state):
#     labelSupports, labelDict = getLabelSupports(np.array(CLASS_LABELS[learningIndices]))
#     nbTrainingExamples = [[int(support / nbFolds) for support in labelSupports] for fold in range(nbFolds)]
#     trainingExamplesIndices = []
#     usedIndices = []
#     for foldIndex, fold in enumerate(nbTrainingExamples):
#         trainingExamplesIndices.append([])
#         while fold != [0 for i in range(NB_CLASS)]:
#             index = random_state.randint(0, len(learningIndices))
#             if learningIndices[index] not in usedIndices:
#                 isUseFull, fold = isUseful(fold, learningIndices[index], CLASS_LABELS, labelDict)
#                 if isUseFull:
#                     trainingExamplesIndices[foldIndex].append(learningIndices[index])
#                     usedIndices.append(learningIndices[index])
#     return trainingExamplesIndices
#
#
# def getPositions(labelsUsed, fullLabels):
#     usedIndices = []
#     for labelIndex, label in enumerate(fullLabels):
#         if label in labelsUsed:
#             usedIndices.append(labelIndex)
#     return usedIndices


# def getCaltechDBcsv(views, pathF, nameDB, NB_CLASS, LABELS_NAMES, random_state):
#     datasetFile = h5py.File(pathF + nameDB + ".hdf5", "w")
#     labelsNamesFile = open(pathF + nameDB + '-ClassLabels-Description.csv')
#     if len(LABELS_NAMES) != NB_CLASS:
#         nbLabelsAvailable = 0
#         for l in labelsNamesFile:
#             nbLabelsAvailable += 1
#         LABELS_NAMES = [line.strip().split(";")[1] for lineIdx, line in enumerate(labelsNamesFile) if
#                         lineIdx in random_state.randint(nbLabelsAvailable, size=NB_CLASS)]
#     fullLabels = np.genfromtxt(pathF + nameDB + '-ClassLabels.csv', delimiter=';').astype(int)
#     labelsDictionary = dict((classIndice, labelName) for (classIndice, labelName) in
#                             [(int(line.strip().split(";")[0]), line.strip().split(";")[1]) for lineIndex, line in
#                              labelsNamesFile if line.strip().split(";")[0] in LABELS_NAMES])
#     if len(set(fullLabels)) > NB_CLASS:
#         usedIndices = getPositions(labelsDictionary.keys(), fullLabels)
#     else:
#         usedIndices = range(len(fullLabels))
#     for viewIndex, view in enumerate(views):
#         viewFile = pathF + nameDB + "-" + view + '.csv'
#         viewMatrix = np.array(np.genfromtxt(viewFile, delimiter=';'))[usedIndices, :]
#         viewDset = datasetFile.create_dataset("View" + str(viewIndex), viewMatrix.shape, data=viewMatrix)
#         viewDset.attrs["name"] = view
#
#     labelsDset = datasetFile.create_dataset("Labels", fullLabels[usedIndices].shape, data=fullLabels[usedIndices])
#
#     metaDataGrp = datasetFile.create_group("Metadata")
#     metaDataGrp.attrs["nbView"] = len(views)
#     metaDataGrp.attrs["nbClass"] = NB_CLASS
#     metaDataGrp.attrs["datasetLength"] = len(fullLabels[usedIndices])
#     datasetFile.close()
#     datasetFile = h5py.File(pathF + nameDB + ".hdf5", "r")
#     return datasetFile, labelsDictionary

# --------------------------------------------#
# All the functions below are not useful     #
# anymore but the binarization methods in    #
# it must be kept                            #
# --------------------------------------------#


# def getMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES, random_state):
#     datasetFile = h5py.File(path + "MultiOmic.hdf5", "w")
#
#     logging.debug("Start:\t Getting Methylation data")
#     methylData = np.genfromtxt(path + "matching_methyl.csv", delimiter=',')
#     methylDset = datasetFile.create_dataset("View0", methylData.shape)
#     methylDset[...] = methylData
#     methylDset.attrs["name"] = "Methyl"
#     methylDset.attrs["sparse"] = False
#     methylDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Methylation data")
#
#     logging.debug("Start:\t Getting MiRNA data")
#     mirnaData = np.genfromtxt(path + "matching_mirna.csv", delimiter=',')
#     mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
#     mirnaDset[...] = mirnaData
#     mirnaDset.attrs["name"] = "MiRNA_"
#     mirnaDset.attrs["sparse"] = False
#     mirnaDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting MiRNA data")
#
#     logging.debug("Start:\t Getting RNASeq data")
#     rnaseqData = np.genfromtxt(path + "matching_rnaseq.csv", delimiter=',')
#     uselessRows = []
#     for rowIndex, row in enumerate(np.transpose(rnaseqData)):
#         if not row.any():
#             uselessRows.append(rowIndex)
#     usefulRows = [usefulRowIndex for usefulRowIndex in range(rnaseqData.shape[1]) if usefulRowIndex not in uselessRows]
#     rnaseqDset = datasetFile.create_dataset("View2", (rnaseqData.shape[0], len(usefulRows)))
#     rnaseqDset[...] = rnaseqData[:, usefulRows]
#     rnaseqDset.attrs["name"] = "RNASeq_"
#     rnaseqDset.attrs["sparse"] = False
#     rnaseqDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting RNASeq data")
#
#     logging.debug("Start:\t Getting Clinical data")
#     clinical = np.genfromtxt(path + "clinicalMatrix.csv", delimiter=',')
#     clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
#     clinicalDset[...] = clinical
#     clinicalDset.attrs["name"] = "Clinic"
#     clinicalDset.attrs["sparse"] = False
#     clinicalDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Clinical data")
#
#     labelFile = open(path + 'brca_labels_triple-negatif.csv')
#     labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
#     labelsDset = datasetFile.create_dataset("Labels", labels.shape)
#     labelsDset[...] = labels
#     labelsDset.attrs["name"] = "Labels"
#
#     metaDataGrp = datasetFile.create_group("Metadata")
#     metaDataGrp.attrs["nbView"] = 4
#     metaDataGrp.attrs["nbClass"] = 2
#     metaDataGrp.attrs["datasetLength"] = len(labels)
#     labelDictionary = {0: "No", 1: "Yes"}
#     datasetFile.close()
#     datasetFile = h5py.File(path + "MultiOmic.hdf5", "r")
#     # datasetFile = getPseudoRNASeq(datasetFile)
#     return datasetFile, labelDictionary
#
#
# def getVector(nbGenes):
#     argmax = [0, 0]
#     maxi = 0
#     for i in range(nbGenes):
#         for j in range(nbGenes):
#             if j == i + 1:
#                 value = (i + 1) * (nbGenes - j)
#                 if value > maxi:
#                     maxi = value
#                     argmax = [i, j]
#     i, j = argmax
#     vectorLeft = np.zeros(nbGenes, dtype=bool)
#     vectorLeft[:i + 1] = np.ones(i + 1, dtype=bool)
#     vectorSup = np.zeros(nbGenes, dtype=bool)
#     vectorSup[j:] = np.ones(nbGenes - j, dtype=bool)
#     matrixSup = j
#     matrixInf = nbGenes - j
#     return vectorLeft, matrixSup, matrixInf
#
#
# def findClosestPowerOfTwo(factorizationParam):
#     power = 1
#     while factorizationParam - power > 0:
#         power *= 2
#     if abs(factorizationParam - power) < abs(factorizationParam - power / 2):
#         return power
#     else:
#         return power / 2
#
#
# def easyFactorize(nbGenes, factorizationParam, t=0):
#     if math.log(factorizationParam + 1, 2) % 1 == 0.0:
#         pass
#     else:
#         factorizationParam = findClosestPowerOfTwo(factorizationParam) - 1
#
#     if nbGenes == 2:
#         return 1, np.array([True, False])
#
#     if nbGenes == 3:
#         return 1, np.array([True, True, False])
#
#     if factorizationParam == 1:
#         t = 1
#         return t, getVector(nbGenes)[0]
#
#     vectorLeft, matrixSup, matrixInf = getVector(nbGenes)
#
#     t_, vectorLeftSup = easyFactorize(matrixSup, (factorizationParam - 1) / 2, t=t)
#     t__, vectorLeftInf = easyFactorize(matrixInf, (factorizationParam - 1) / 2, t=t)
#
#     factorLeft = np.zeros((nbGenes, t_ + t__ + 1), dtype=bool)
#
#     factorLeft[:matrixSup, :t_] = vectorLeftSup.reshape(factorLeft[:matrixSup, :t_].shape)
#     if nbGenes % 2 == 1:
#         factorLeft[matrixInf - 1:, t_:t__ + t_] = vectorLeftInf.reshape(factorLeft[matrixInf - 1:, t_:t__ + t_].shape)
#     else:
#         factorLeft[matrixInf:, t_:t__ + t_] = vectorLeftInf.reshape(factorLeft[matrixInf:, t_:t__ + t_].shape)
#     factorLeft[:, t__ + t_] = vectorLeft
#
#     # factorSup = np.zeros((t_+t__+1, nbGenes), dtype=bool)
#     #
#     # factorSup[:t_, :matrixSup] = vectorSupLeft.reshape(factorSup[:t_, :matrixSup].shape)
#     # if nbGenes%2==1:
#     #     factorSup[t_:t__+t_, matrixInf-1:] = vectorSupRight.reshape(factorSup[t_:t__+t_, matrixInf-1:].shape)
#     # else:
#     #     factorSup[t_:t__+t_, matrixInf:] = vectorSupRight.reshape(factorSup[t_:t__+t_, matrixInf:].shape)
#     # factorSup[t__+t_, :] = vectorSup
#     return t__ + t_ + 1, factorLeft  # , factorSup
#
#
# def getBaseMatrices(nbGenes, factorizationParam, path):
#     t, factorLeft = easyFactorize(nbGenes, factorizationParam)
#     np.savetxt(path + "factorLeft--n-" + str(nbGenes) + "--k-" + str(factorizationParam) + ".csv", factorLeft,
#                delimiter=",")
#     return factorLeft
#
#
# def findParams(arrayLen, nbPatients, random_state, maxNbBins=2000, minNbBins=10, maxLenBin=70000, minOverlapping=1,
#                minNbBinsOverlapped=0, maxNbSolutions=30):
#     results = []
#     if arrayLen * arrayLen * 10 / 100 > minNbBinsOverlapped * nbPatients:
#         for lenBin in range(arrayLen - 1):
#             lenBin += 1
#             if lenBin < maxLenBin and minNbBins * lenBin < arrayLen:
#                 for overlapping in sorted(range(lenBin - 1), reverse=True):
#                     overlapping += 1
#                     if overlapping > minOverlapping and lenBin % (lenBin - overlapping) == 0:
#                         for nbBins in sorted(range(arrayLen - 1), reverse=True):
#                             nbBins += 1
#                             if nbBins < maxNbBins:
#                                 if arrayLen == (nbBins - 1) * (lenBin - overlapping) + lenBin:
#                                     results.append({"nbBins": nbBins, "overlapping": overlapping, "lenBin": lenBin})
#                                     if len(results) == maxNbSolutions:
#                                         params = preds[random_state.randrange(len(preds))]
#                                         return params
#
#
# def findBins(nbBins=142, overlapping=493, lenBin=986):
#     bins = []
#     for binIndex in range(nbBins):
#         bins.append([i + binIndex * (lenBin - overlapping) for i in range(lenBin)])
#     return bins
#
#
# def getBins(array, bins, lenBin, overlapping):
#     binnedcoord = []
#     for coordIndex, coord in enumerate(array):
#         nbBinsFull = 0
#         for binIndex, bin_ in enumerate(bins):
#             if coordIndex in bin_:
#                 binnedcoord.append(binIndex + (coord * len(bins)))
#
#     return np.array(binnedcoord)
#
#
# def makeSortedBinsMatrix(nbBins, lenBins, overlapping, arrayLen, path):
#     sortedBinsMatrix = np.zeros((arrayLen, nbBins), dtype=np.uint8)
#     step = lenBins - overlapping
#     for binIndex in range(nbBins):
#         sortedBinsMatrix[step * binIndex:lenBins + (step * binIndex), binIndex] = np.ones(lenBins, dtype=np.uint8)
#     np.savetxt(path + "sortedBinsMatrix--t-" + str(lenBins) + "--n-" + str(nbBins) + "--c-" + str(overlapping) + ".csv",
#                sortedBinsMatrix, delimiter=",")
#     return sortedBinsMatrix
#
#
# def makeSparseTotalMatrix(sortedRNASeq, random_state):
#     nbPatients, nbGenes = sortedRNASeq.shape
#     params = findParams(nbGenes, nbPatients, random_state)
#     nbBins = params["nbBins"]
#     overlapping = params["overlapping"]
#     lenBin = params["lenBin"]
#     bins = findBins(nbBins, overlapping, lenBin)
#     sparseFull = sparse.csc_matrix((nbPatients, nbGenes * nbBins))
#     for patientIndex, patient in enumerate(sortedRNASeq):
#         columnIndices = getBins(patient, bins, lenBin, overlapping)
#         rowIndices = np.zeros(len(columnIndices), dtype=int) + patientIndex
#         data = np.ones(len(columnIndices), dtype=bool)
#         sparseFull = sparseFull + sparse.csc_matrix((data, (rowIndices, columnIndices)),
#                                                     shape=(nbPatients, nbGenes * nbBins))
#     return sparseFull
#
#
# def getAdjacenceMatrix(RNASeqRanking, sotredRNASeq, k=2):
#     k = int(k) / 2 * 2
#     indices = np.zeros((RNASeqRanking.shape[0] * k * RNASeqRanking.shape[1]), dtype=int)
#     data = np.ones((RNASeqRanking.shape[0] * k * RNASeqRanking.shape[1]), dtype=bool)
#     indptr = np.zeros(RNASeqRanking.shape[0] + 1, dtype=int)
#     nbGenes = RNASeqRanking.shape[1]
#     pointer = 0
#     for patientIndex in range(RNASeqRanking.shape[0]):
#         for i in range(nbGenes):
#             for j in range(k / 2):
#                 try:
#                     indices[pointer] = RNASeqRanking[
#                                            patientIndex, (sotredRNASeq[patientIndex, i] - (j + 1))] + i * nbGenes
#                     pointer += 1
#                 except:
#                     pass
#                 try:
#                     indices[pointer] = RNASeqRanking[
#                                            patientIndex, (sotredRNASeq[patientIndex, i] + (j + 1))] + i * nbGenes
#                     pointer += 1
#                 except:
#                     pass
#                     # elif i<=k:
#                     # 	indices.append(patient[1]+patient[i]*nbGenes)
#                     # 	data.append(True)
#                     # elif i==nbGenes-1:
#                     # 	indices.append(patient[i-1]+patient[i]*nbGenes)
#                     # 	data.append(True)
#         indptr[patientIndex + 1] = pointer
#
#     mat = sparse.csr_matrix((data, indices, indptr),
#                             shape=(RNASeqRanking.shape[0], RNASeqRanking.shape[1] * RNASeqRanking.shape[1]), dtype=bool)
#     return mat
#
#
# def getKMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):
#     datasetFile = h5py.File(path + "KMultiOmic.hdf5", "w")
#
#     # logging.debug("Start:\t Getting Methylation data")
#     methylData = np.genfromtxt(path + "matching_methyl.csv", delimiter=',')
#     logging.debug("Done:\t Getting Methylation data")
#
#     logging.debug("Start:\t Getting Sorted Methyl data")
#     Methyl = methylData
#     sortedMethylGeneIndices = np.zeros(methylData.shape, dtype=int)
#     MethylRanking = np.zeros(methylData.shape, dtype=int)
#     for sampleIndex, sampleArray in enumerate(Methyl):
#         sortedMethylDictionary = dict((index, value) for index, value in enumerate(sampleArray))
#         sortedMethylIndicesDict = sorted(sortedMethylDictionary.items(), key=operator.itemgetter(1))
#         sortedMethylIndicesArray = np.array([index for (index, value) in sortedMethylIndicesDict], dtype=int)
#         sortedMethylGeneIndices[sampleIndex] = sortedMethylIndicesArray
#         for geneIndex in range(Methyl.shape[1]):
#             MethylRanking[sampleIndex, sortedMethylIndicesArray[geneIndex]] = geneIndex
#     logging.debug("Done:\t Getting Sorted Methyl data")
#
#     logging.debug("Start:\t Getting Binarized Methyl data")
#     k = findClosestPowerOfTwo(9) - 1
#     try:
#         factorizedLeftBaseMatrix = np.genfromtxt(
#             path + "factorLeft--n-" + str(methylData.shape[1]) + "--k-" + str(k) + ".csv", delimiter=',')
#     except:
#         factorizedLeftBaseMatrix = getBaseMatrices(methylData.shape[1], k, path)
#     bMethylDset = datasetFile.create_dataset("View0",
#                                              (sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1] * k),
#                                              dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
#         patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], k), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = factorizedLeftBaseMatrix[lineIndex, :]
#         bMethylDset[patientIndex] = patientMatrix.flatten()
#     bMethylDset.attrs["name"] = "BMethyl" + str(k)
#     bMethylDset.attrs["sparse"] = False
#     bMethylDset.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binarized Methyl data")
#
#     logging.debug("Start:\t Getting Binned Methyl data")
#     lenBins = 3298
#     nbBins = 9
#     overlapping = 463
#     try:
#         sortedBinsMatrix = np.genfromtxt(
#             path + "sortedBinsMatrix--t-" + str(lenBins) + "--n-" + str(nbBins) + "--c-" + str(overlapping) + ".csv",
#             delimiter=",")
#     except:
#         sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, methylData.shape[1], path)
#     binnedMethyl = datasetFile.create_dataset("View1", (
#         sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1] * nbBins), dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
#         patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], nbBins), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = sortedBinsMatrix[lineIndex, :]
#         binnedMethyl[patientIndex] = patientMatrix.flatten()
#     binnedMethyl.attrs["name"] = "bMethyl" + str(nbBins)
#     binnedMethyl.attrs["sparse"] = False
#     binnedMethyl.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binned Methyl data")
#
#     logging.debug("Start:\t Getting Binarized Methyl data")
#     k = findClosestPowerOfTwo(17) - 1
#     try:
#         factorizedLeftBaseMatrix = np.genfromtxt(
#             path + "factorLeft--n-" + str(methylData.shape[1]) + "--k-" + str(k) + ".csv", delimiter=',')
#     except:
#         factorizedLeftBaseMatrix = getBaseMatrices(methylData.shape[1], k, path)
#     bMethylDset = datasetFile.create_dataset("View2",
#                                              (sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1] * k),
#                                              dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
#         patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], k), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = factorizedLeftBaseMatrix[lineIndex, :]
#         bMethylDset[patientIndex] = patientMatrix.flatten()
#     bMethylDset.attrs["name"] = "BMethyl" + str(k)
#     bMethylDset.attrs["sparse"] = False
#     bMethylDset.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binarized Methyl data")
#
#     logging.debug("Start:\t Getting Binned Methyl data")
#     lenBins = 2038
#     nbBins = 16
#     overlapping = 442
#     try:
#         sortedBinsMatrix = np.genfromtxt(
#             path + "sortedBinsMatrix--t-" + str(lenBins) + "--n-" + str(nbBins) + "--c-" + str(overlapping) + ".csv",
#             delimiter=",")
#     except:
#         sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, methylData.shape[1], path)
#     binnedMethyl = datasetFile.create_dataset("View3", (
#         sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1] * nbBins), dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
#         patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], nbBins), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = sortedBinsMatrix[lineIndex, :]
#         binnedMethyl[patientIndex] = patientMatrix.flatten()
#     binnedMethyl.attrs["name"] = "bMethyl" + str(nbBins)
#     binnedMethyl.attrs["sparse"] = False
#     binnedMethyl.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binned Methyl data")
#
#     labelFile = open(path + 'brca_labels_triple-negatif.csv')
#     labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
#     labelsDset = datasetFile.create_dataset("Labels", labels.shape)
#     labelsDset[...] = labels
#     labelsDset.attrs["name"] = "Labels"
#
#     metaDataGrp = datasetFile.create_group("Metadata")
#     metaDataGrp.attrs["nbView"] = 4
#     metaDataGrp.attrs["nbClass"] = 2
#     metaDataGrp.attrs["datasetLength"] = len(labels)
#     labelDictionary = {0: "No", 1: "Yes"}
#
#     datasetFile.close()
#     datasetFile = h5py.File(path + "KMultiOmic.hdf5", "r")
#
#     return datasetFile, labelDictionary
#
#
# def getKMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
#     datasetFile = h5py.File(path + "KMultiOmic.hdf5", "r")
#     labelDictionary = {0: "No", 1: "Yes"}
#     return datasetFile, labelDictionary
#
#
# def getModifiedMultiOmicDBcsv(features, path, name, NB_CLASS, LABELS_NAMES):
#     datasetFile = h5py.File(path + "ModifiedMultiOmic.hdf5", "w")
#
#     logging.debug("Start:\t Getting Methylation data")
#     methylData = np.genfromtxt(path + "matching_methyl.csv", delimiter=',')
#     methylDset = datasetFile.create_dataset("View0", methylData.shape)
#     methylDset[...] = methylData
#     methylDset.attrs["name"] = "Methyl_"
#     methylDset.attrs["sparse"] = False
#     methylDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Methylation data")
#
#     logging.debug("Start:\t Getting Sorted Methyl data")
#     Methyl = datasetFile["View0"][...]
#     sortedMethylGeneIndices = np.zeros(datasetFile.get("View0").shape, dtype=int)
#     MethylRanking = np.zeros(datasetFile.get("View0").shape, dtype=int)
#     for sampleIndex, sampleArray in enumerate(Methyl):
#         sortedMethylDictionary = dict((index, value) for index, value in enumerate(sampleArray))
#         sortedMethylIndicesDict = sorted(sortedMethylDictionary.items(), key=operator.itemgetter(1))
#         sortedMethylIndicesArray = np.array([index for (index, value) in sortedMethylIndicesDict], dtype=int)
#         sortedMethylGeneIndices[sampleIndex] = sortedMethylIndicesArray
#         for geneIndex in range(Methyl.shape[1]):
#             MethylRanking[sampleIndex, sortedMethylIndicesArray[geneIndex]] = geneIndex
#     mMethylDset = datasetFile.create_dataset("View10", sortedMethylGeneIndices.shape, data=sortedMethylGeneIndices)
#     mMethylDset.attrs["name"] = "SMethyl"
#     mMethylDset.attrs["sparse"] = False
#     mMethylDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Sorted Methyl data")
#
#     logging.debug("Start:\t Getting Binarized Methyl data")
#     k = findClosestPowerOfTwo(58) - 1
#     try:
#         factorizedLeftBaseMatrix = np.genfromtxt(
#             path + "factorLeft--n-" + str(datasetFile.get("View0").shape[1]) + "--k-" + str(k) + ".csv", delimiter=',')
#     except:
#         factorizedLeftBaseMatrix = getBaseMatrices(methylData.shape[1], k, path)
#     bMethylDset = datasetFile.create_dataset("View11",
#                                              (sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1] * k),
#                                              dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
#         patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], k), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = factorizedLeftBaseMatrix[lineIndex, :]
#         bMethylDset[patientIndex] = patientMatrix.flatten()
#     bMethylDset.attrs["name"] = "BMethyl"
#     bMethylDset.attrs["sparse"] = False
#     bMethylDset.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binarized Methyl data")
#
#     logging.debug("Start:\t Getting Binned Methyl data")
#     lenBins = 2095
#     nbBins = 58
#     overlapping = 1676
#     try:
#         sortedBinsMatrix = np.genfromtxt(
#             path + "sortedBinsMatrix--t-" + str(lenBins) + "--n-" + str(nbBins) + "--c-" + str(overlapping) + ".csv",
#             delimiter=",")
#     except:
#         sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, datasetFile.get("View0").shape[1], path)
#     binnedMethyl = datasetFile.create_dataset("View12", (
#         sortedMethylGeneIndices.shape[0], sortedMethylGeneIndices.shape[1] * nbBins), dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMethylGeneIndices):
#         patientMatrix = np.zeros((sortedMethylGeneIndices.shape[1], nbBins), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = sortedBinsMatrix[lineIndex, :]
#         binnedMethyl[patientIndex] = patientMatrix.flatten()
#     binnedMethyl.attrs["name"] = "bMethyl"
#     binnedMethyl.attrs["sparse"] = False
#     binnedMethyl.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binned Methyl data")
#
#     logging.debug("Start:\t Getting MiRNA data")
#     mirnaData = np.genfromtxt(path + "matching_mirna.csv", delimiter=',')
#     mirnaDset = datasetFile.create_dataset("View1", mirnaData.shape)
#     mirnaDset[...] = mirnaData
#     mirnaDset.attrs["name"] = "MiRNA__"
#     mirnaDset.attrs["sparse"] = False
#     mirnaDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting MiRNA data")
#
#     logging.debug("Start:\t Getting Sorted MiRNA data")
#     MiRNA = datasetFile["View1"][...]
#     sortedMiRNAGeneIndices = np.zeros(datasetFile.get("View1").shape, dtype=int)
#     MiRNARanking = np.zeros(datasetFile.get("View1").shape, dtype=int)
#     for sampleIndex, sampleArray in enumerate(MiRNA):
#         sortedMiRNADictionary = dict((index, value) for index, value in enumerate(sampleArray))
#         sortedMiRNAIndicesDict = sorted(sortedMiRNADictionary.items(), key=operator.itemgetter(1))
#         sortedMiRNAIndicesArray = np.array([index for (index, value) in sortedMiRNAIndicesDict], dtype=int)
#         sortedMiRNAGeneIndices[sampleIndex] = sortedMiRNAIndicesArray
#         for geneIndex in range(MiRNA.shape[1]):
#             MiRNARanking[sampleIndex, sortedMiRNAIndicesArray[geneIndex]] = geneIndex
#     mmirnaDset = datasetFile.create_dataset("View7", sortedMiRNAGeneIndices.shape, data=sortedMiRNAGeneIndices)
#     mmirnaDset.attrs["name"] = "SMiRNA_"
#     mmirnaDset.attrs["sparse"] = False
#     mmirnaDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Sorted MiRNA data")
#
#     logging.debug("Start:\t Getting Binarized MiRNA data")
#     k = findClosestPowerOfTwo(517) - 1
#     try:
#         factorizedLeftBaseMatrix = np.genfromtxt(
#             path + "factorLeft--n-" + str(datasetFile.get("View1").shape[1]) + "--k-" + str(k) + ".csv", delimiter=',')
#     except:
#         factorizedLeftBaseMatrix = getBaseMatrices(mirnaData.shape[1], k, path)
#     bmirnaDset = datasetFile.create_dataset("View8",
#                                             (sortedMiRNAGeneIndices.shape[0], sortedMiRNAGeneIndices.shape[1] * k),
#                                             dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMiRNAGeneIndices):
#         patientMatrix = np.zeros((sortedMiRNAGeneIndices.shape[1], k), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = factorizedLeftBaseMatrix[lineIndex, :]
#         bmirnaDset[patientIndex] = patientMatrix.flatten()
#     bmirnaDset.attrs["name"] = "BMiRNA_"
#     bmirnaDset.attrs["sparse"] = False
#     bmirnaDset.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binarized MiRNA data")
#
#     logging.debug("Start:\t Getting Binned MiRNA data")
#     lenBins = 14
#     nbBins = 517
#     overlapping = 12
#     try:
#         sortedBinsMatrix = np.genfromtxt(
#             path + "sortedBinsMatrix--t-" + str(lenBins) + "--n-" + str(nbBins) + "--c-" + str(overlapping) + ".csv",
#             delimiter=",")
#     except:
#         sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, datasetFile.get("View1").shape[1], path)
#     binnedMiRNA = datasetFile.create_dataset("View9", (
#         sortedMiRNAGeneIndices.shape[0], sortedMiRNAGeneIndices.shape[1] * nbBins), dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedMiRNAGeneIndices):
#         patientMatrix = np.zeros((sortedMiRNAGeneIndices.shape[1], nbBins), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = sortedBinsMatrix[lineIndex, :]
#         binnedMiRNA[patientIndex] = patientMatrix.flatten()
#     binnedMiRNA.attrs["name"] = "bMiRNA_"
#     binnedMiRNA.attrs["sparse"] = False
#     binnedMiRNA.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binned MiRNA data")
#
#     logging.debug("Start:\t Getting RNASeq data")
#     rnaseqData = np.genfromtxt(path + "matching_rnaseq.csv", delimiter=',')
#     uselessRows = []
#     for rowIndex, row in enumerate(np.transpose(rnaseqData)):
#         if not row.any():
#             uselessRows.append(rowIndex)
#     usefulRows = [usefulRowIndex for usefulRowIndex in range(rnaseqData.shape[1]) if usefulRowIndex not in uselessRows]
#     rnaseqDset = datasetFile.create_dataset("View2", (rnaseqData.shape[0], len(usefulRows)))
#     rnaseqDset[...] = rnaseqData[:, usefulRows]
#     rnaseqDset.attrs["name"] = "RNASeq_"
#     rnaseqDset.attrs["sparse"] = False
#     rnaseqDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting RNASeq data")
#
#     logging.debug("Start:\t Getting Sorted RNASeq data")
#     RNASeq = datasetFile["View2"][...]
#     sortedRNASeqGeneIndices = np.zeros(datasetFile.get("View2").shape, dtype=int)
#     RNASeqRanking = np.zeros(datasetFile.get("View2").shape, dtype=int)
#     for sampleIndex, sampleArray in enumerate(RNASeq):
#         sortedRNASeqDictionary = dict((index, value) for index, value in enumerate(sampleArray))
#         sortedRNASeqIndicesDict = sorted(sortedRNASeqDictionary.items(), key=operator.itemgetter(1))
#         sortedRNASeqIndicesArray = np.array([index for (index, value) in sortedRNASeqIndicesDict], dtype=int)
#         sortedRNASeqGeneIndices[sampleIndex] = sortedRNASeqIndicesArray
#         for geneIndex in range(RNASeq.shape[1]):
#             RNASeqRanking[sampleIndex, sortedRNASeqIndicesArray[geneIndex]] = geneIndex
#     mrnaseqDset = datasetFile.create_dataset("View4", sortedRNASeqGeneIndices.shape, data=sortedRNASeqGeneIndices)
#     mrnaseqDset.attrs["name"] = "SRNASeq"
#     mrnaseqDset.attrs["sparse"] = False
#     mrnaseqDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Sorted RNASeq data")
#
#     logging.debug("Start:\t Getting Binarized RNASeq data")
#     k = findClosestPowerOfTwo(100) - 1
#     try:
#         factorizedLeftBaseMatrix = np.genfromtxt(
#             path + "factorLeft--n-" + str(datasetFile.get("View2").shape[1]) + "--k-" + str(100) + ".csv",
#             delimiter=',')
#     except:
#         factorizedLeftBaseMatrix = getBaseMatrices(rnaseqData.shape[1], k, path)
#     brnaseqDset = datasetFile.create_dataset("View5",
#                                              (sortedRNASeqGeneIndices.shape[0], sortedRNASeqGeneIndices.shape[1] * k),
#                                              dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedRNASeqGeneIndices):
#         patientMatrix = np.zeros((sortedRNASeqGeneIndices.shape[1], k), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = factorizedLeftBaseMatrix[lineIndex, :]
#         brnaseqDset[patientIndex] = patientMatrix.flatten()
#     brnaseqDset.attrs["name"] = "BRNASeq"
#     brnaseqDset.attrs["sparse"] = False
#     brnaseqDset.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binarized RNASeq data")
#
#     logging.debug("Start:\t Getting Binned RNASeq data")
#     lenBins = 986
#     nbBins = 142
#     overlapping = 493
#     try:
#         sortedBinsMatrix = np.genfromtxt(
#             path + "sortedBinsMatrix--t-" + str(lenBins) + "--n-" + str(nbBins) + "--c-" + str(overlapping) + ".csv",
#             delimiter=",")
#     except:
#         sortedBinsMatrix = makeSortedBinsMatrix(nbBins, lenBins, overlapping, datasetFile.get("View2").shape[1], path)
#     binnedRNASeq = datasetFile.create_dataset("View6", (
#         sortedRNASeqGeneIndices.shape[0], sortedRNASeqGeneIndices.shape[1] * nbBins), dtype=np.uint8)
#     for patientIndex, patientSortedArray in enumerate(sortedRNASeqGeneIndices):
#         patientMatrix = np.zeros((sortedRNASeqGeneIndices.shape[1], nbBins), dtype=np.uint8)
#         for lineIndex, geneIndex in enumerate(patientSortedArray):
#             patientMatrix[geneIndex] = sortedBinsMatrix[lineIndex, :]
#         binnedRNASeq[patientIndex] = patientMatrix.flatten()
#     binnedRNASeq.attrs["name"] = "bRNASeq"
#     binnedRNASeq.attrs["sparse"] = False
#     binnedRNASeq.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binned RNASeq data")
#
#     logging.debug("Start:\t Getting Clinical data")
#     clinical = np.genfromtxt(path + "clinicalMatrix.csv", delimiter=',')
#     clinicalDset = datasetFile.create_dataset("View3", clinical.shape)
#     clinicalDset[...] = clinical
#     clinicalDset.attrs["name"] = "Clinic_"
#     clinicalDset.attrs["sparse"] = False
#     clinicalDset.attrs["binary"] = False
#     logging.debug("Done:\t Getting Clinical data")
#
#     logging.debug("Start:\t Getting Binarized Clinical data")
#     binarized_clinical = np.zeros((347, 1951), dtype=np.uint8)
#     nb_already_done = 0
#     for feqtureIndex, feature in enumerate(np.transpose(clinical)):
#         featureSet = set(feature)
#         featureDict = dict((val, valIndex) for valIndex, val in enumerate(list(featureSet)))
#         for valueIndex, value in enumerate(feature):
#             binarized_clinical[valueIndex, featureDict[value] + nb_already_done] = 1
#         nb_already_done += len(featureSet)
#     bClinicalDset = datasetFile.create_dataset("View13", binarized_clinical.shape, dtype=np.uint8,
#                                                data=binarized_clinical)
#     bClinicalDset.attrs["name"] = "bClinic"
#     bClinicalDset.attrs["sparse"] = False
#     bClinicalDset.attrs["binary"] = True
#     logging.debug("Done:\t Getting Binarized Clinical data")
#
#     # logging.debug("Start:\t Getting Adjacence RNASeq data")
#     # sparseAdjRNASeq = getAdjacenceMatrix(RNASeqRanking, sortedRNASeqGeneIndices, k=findClosestPowerOfTwo(10)-1)
#     # sparseAdjRNASeqGrp = datasetFile.create_group("View6")
#     # dataDset = sparseAdjRNASeqGrp.create_dataset("data", sparseAdjRNASeq.data.shape, data=sparseAdjRNASeq.data)
#     # indicesDset = sparseAdjRNASeqGrp.create_dataset("indices",
#     # sparseAdjRNASeq.indices.shape, data=sparseAdjRNASeq.indices)
#     # indptrDset = sparseAdjRNASeqGrp.create_dataset("indptr",
#     # sparseAdjRNASeq.indptr.shape, data=sparseAdjRNASeq.indptr)
#     # sparseAdjRNASeqGrp.attrs["name"]="ARNASeq"
#     # sparseAdjRNASeqGrp.attrs["sparse"]=True
#     # sparseAdjRNASeqGrp.attrs["shape"]=sparseAdjRNASeq.shape
#     # logging.debug("Done:\t Getting Adjacence RNASeq data")
#
#     labelFile = open(path + 'brca_labels_triple-negatif.csv')
#     labels = np.array([int(line.strip().split(',')[1]) for line in labelFile])
#     labelsDset = datasetFile.create_dataset("Labels", labels.shape)
#     labelsDset[...] = labels
#     labelsDset.attrs["name"] = "Labels"
#
#     metaDataGrp = datasetFile.create_group("Metadata")
#     metaDataGrp.attrs["nbView"] = 14
#     metaDataGrp.attrs["nbClass"] = 2
#     metaDataGrp.attrs["datasetLength"] = len(labels)
#     labelDictionary = {0: "No", 1: "Yes"}
#
#     datasetFile.close()
#     datasetFile = h5py.File(path + "ModifiedMultiOmic.hdf5", "r")
#
#     return datasetFile, labelDictionary
#
#
# def getModifiedMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
#     datasetFile = h5py.File(path + "ModifiedMultiOmic.hdf5", "r")
#     labelDictionary = {0: "No", 1: "Yes"}
#     return datasetFile, labelDictionary
#
#
# def getMultiOmicDBhdf5(features, path, name, NB_CLASS, LABELS_NAMES):
#     datasetFile = h5py.File(path + "MultiOmic.hdf5", "r")
#     labelDictionary = {0: "No", 1: "Yes"}
#     return datasetFile, labelDictionary
#
#
#
# # def getOneViewFromDB(viewName, pathToDB, DBName):
# #     view = np.genfromtxt(pathToDB + DBName +"-" + viewName, delimiter=';')
# #     return view
#
#
# # def getClassLabels(pathToDB, DBName):
# #     labels = np.genfromtxt(pathToDB + DBName + "-" + "ClassLabels.csv", delimiter=';')
# #     return labels
#
#
# # def getDataset(pathToDB, viewNames, DBName):
# #     dataset = []
# #     for viewName in viewNames:
# #         dataset.append(getOneViewFromDB(viewName, pathToDB, DBName))
# #     return np.array(dataset)
#
#
# # def getAwaLabels(nbLabels, pathToAwa):
# #     labelsFile = open(pathToAwa + 'Animals_with_Attributes/classes.txt', 'U')
# #     linesFile = [''.join(line.strip().split()).translate(None, digits) for line in labelsFile.readlines()]
# #     return linesFile
#
#
# # def getAwaDBcsv(views, pathToAwa, nameDB, nbLabels, LABELS_NAMES):
# #     awaLabels = getAwaLabels(nbLabels, pathToAwa)
# #     nbView = len(views)
# #     nbMaxLabels = len(awaLabels)
# #     if nbLabels == -1:
# #         nbLabels = nbMaxLabels
# #     nbNamesGiven = len(LABELS_NAMES)
# #     if nbNamesGiven > nbLabels:
# #         labelDictionary = {i:LABELS_NAMES[i] for i in np.arange(nbLabels)}
# #     elif nbNamesGiven < nbLabels and nbLabels <= nbMaxLabels:
# #         if LABELS_NAMES != ['']:
# #             labelDictionary = {i:LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
# #         else:
# #             labelDictionary = {}
# #             nbNamesGiven = 0
# #         nbLabelsToAdd = nbLabels-nbNamesGiven
# #         while nbLabelsToAdd > 0:
# #             currentLabel = random.choice(awaLabels)
# #             if currentLabel not in labelDictionary.values():
# #                 labelDictionary[nbLabels-nbLabelsToAdd]=currentLabel
# #                 nbLabelsToAdd -= 1
# #             else:
# #                 pass
# #     else:
# #         labelDictionary = {i: LABELS_NAMES[i] for i in np.arange(nbNamesGiven)}
# #     viewDictionary = {i: views[i] for i in np.arange(nbView)}
# #     rawData = []
# #     labels = []
# #     nbExample = 0
# #     for view in np.arange(nbView):
# #         viewData = []
# #         for labelIndex in np.arange(nbLabels):
# #             pathToExamples = pathToAwa + 'Animals_with_Attributes/Features/' + viewDictionary[view] + '/' + \
# #                              labelDictionary[labelIndex] + '/'
# #             samples = os.listdir(pathToExamples)
# #             if view == 0:
# #                 nbExample += len(samples)
# #             for sample in samples:
# #                 if viewDictionary[view]=='decaf':
# #                     sampleFile = open(pathToExamples + sample)
# #                     viewData.append([float(line.strip()) for line in sampleFile])
# #                 else:
# #                     sampleFile = open(pathToExamples + sample)
# #                     viewData.append([[float(coordinate) for coordinate in raw.split()] for raw in sampleFile][0])
# #                 if view == 0:
# #                     labels.append(labelIndex)
# #
# #         rawData.append(np.array(viewData))
# #     data = rawData
# #     DATASET_LENGTH = len(labels)
# #     return data, labels, labelDictionary, DATASET_LENGTH
# #
# #
# # def getDbfromCSV(path):
# #     files = os.listdir(path)
# #     DATA = np.zeros((3,40,2))
# #     for file in files:
# #         if file[-9:]=='moins.csv' and file[:7]=='sample1':
# #             X = open(path+file)
# #             for x, i in zip(X, range(20)):
# #                 DATA[0, i] = np.array([float(coord) for coord in x.strip().split('\t')])
# #         if file[-9:]=='moins.csv' and file[:7]=='sample2':
# #             X = open(path+file)
# #             for x, i in zip(X, range(20)):
# #                 DATA[1, i] = np.array([float(coord) for coord in x.strip().split('\t')])
# #         if file[-9:]=='moins.csv' and file[:7]=='sample3':
# #             X = open(path+file)
# #             for x, i in zip(X, range(20)):
# #                 DATA[2, i] = np.array([float(coord) for coord in x.strip().split('\t')])
# #
# #     for file in files:
# #         if file[-8:]=='plus.csv' and file[:7]=='sample1':
# #             X = open(path+file)
# #             for x, i in zip(X, range(20)):
# #                 DATA[0, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
# #         if file[-8:]=='plus.csv' and file[:7]=='sample2':
# #             X = open(path+file)
# #             for x, i in zip(X, range(20)):
# #                 DATA[1, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
# #         if file[-8:]=='plus.csv' and file[:7]=='sample3':
# #             X = open(path+file)
# #             for x, i in zip(X, range(20)):
# #                 DATA[2, i+20] = np.array([float(coord) for coord in x.strip().split('\t')])
# #     LABELS = np.zeros(40)
# #     LABELS[:20]=LABELS[:20]+1
# #     return DATA, LABELS
#
# # def makeArrayFromTriangular(pseudoRNASeqMatrix):
# #     matrixShape = len(pseudoRNASeqMatrix[0,:])
# #     sampleArray = np.array(((matrixShape-1)*matrixShape)/2)
# #     arrayIndex = 0
# #     for i in range(matrixShape-1):
# #         for j in range(i+1, matrixShape):
# #             sampleArray[arrayIndex]=pseudoRNASeqMatrix[i,j]
# #             arrayIndex += 1
# #     return sampleArray
#
#
# # def getPseudoRNASeq(dataset):
# #     nbGenes = len(dataset["/View2/matrix"][0, :])
# #     pseudoRNASeq = np.zeros((dataset["/datasetlength"][...], ((nbGenes - 1) * nbGenes) / 2), dtype=bool_)
# #     for sampleIndex in xrange(dataset["/datasetlength"][...]):
# #         arrayIndex = 0
# #         for i in xrange(nbGenes):
# #             for j in xrange(nbGenes):
# #                 if i > j:
# #                     pseudoRNASeq[sampleIndex, arrayIndex] =
# # dataset["/View2/matrix"][sampleIndex, j] < dataset["/View2/matrix"][sampleIndex, i]
# #                     arrayIndex += 1
# #     dataset["/View4/matrix"] = pseudoRNASeq
# #     dataset["/View4/name"] = "pseudoRNASeq"
# #     return dataset
#
#
# # def allSame(array):
# #     value = array[0]
# #     areAllSame = True
# #     for i in array:
# #         if i != value:
# #             areAllSame = False
# #     return areAllSame


# def getFakeDBhdf5(features, pathF, name, NB_CLASS, LABELS_NAME, random_state):
#     """Was used to generateafake dataset to run tests"""
#     NB_VIEW = 4
#     DATASET_LENGTH = 30
#     NB_CLASS = 2
#     VIEW_DIMENSIONS = random_state.random_integers(5, 20, NB_VIEW)
#
#     DATA = dict((indx,
#                  np.array([
#                               random_state.normal(0.0, 2, viewDimension)
#                               for i in np.arange(DATASET_LENGTH)]))
#                 for indx, viewDimension in enumerate(VIEW_DIMENSIONS))
#
#     CLASS_LABELS = random_state.random_integers(0, NB_CLASS - 1, DATASET_LENGTH)
#     datasetFile = h5py.File(pathF + "Fake.hdf5", "w")
#     for index, viewData in enumerate(DATA.values()):
#         if index == 0:
#             viewData = random_state.randint(0, 1, (DATASET_LENGTH, 300)).astype(
#                 np.uint8)
#             # np.zeros(viewData.shape, dtype=bool)+np.ones((viewData.shape[0], viewData.shape[1]/2), dtype=bool)
#             viewDset = datasetFile.create_dataset("View" + str(index), viewData.shape)
#             viewDset[...] = viewData
#             viewDset.attrs["name"] = "View" + str(index)
#             viewDset.attrs["sparse"] = False
#         elif index == 1:
#             viewData = sparse.csr_matrix(viewData)
#             viewGrp = datasetFile.create_group("View" + str(index))
#             dataDset = viewGrp.create_dataset("data", viewData.data.shape, data=viewData.data)
#             indicesDset = viewGrp.create_dataset("indices", viewData.indices.shape, data=viewData.indices)
#             indptrDset = viewGrp.create_dataset("indptr", viewData.indptr.shape, data=viewData.indptr)
#             viewGrp.attrs["name"] = "View" + str(index)
#             viewGrp.attrs["sparse"] = True
#             viewGrp.attrs["shape"] = viewData.shape
#         else:
#             viewDset = datasetFile.create_dataset("View" + str(index), viewData.shape)
#             viewDset[...] = viewData
#             viewDset.attrs["name"] = "View" + str(index)
#             viewDset.attrs["sparse"] = False
#     labelsDset = datasetFile.create_dataset("Labels", CLASS_LABELS.shape)
#     labelsDset[...] = CLASS_LABELS
#     labelsDset.attrs["name"] = "Labels"
#
#     metaDataGrp = datasetFile.create_group("Metadata")
#     metaDataGrp.attrs["nbView"] = NB_VIEW
#     metaDataGrp.attrs["nbClass"] = NB_CLASS
#     metaDataGrp.attrs["datasetLength"] = len(CLASS_LABELS)
#     labels_dictionary = {0: "No", 1: "Yes"}
#     datasetFile.close()
#     datasetFile = h5py.File(pathF + "Fake.hdf5", "r")
#     return datasetFile, labels_dictionary
