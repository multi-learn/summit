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
import select
import sys
from abc import abstractmethod

import h5py
import numpy as np

from .organization import secure_file_path

'''This is the multiview dataset module. It garthers all the method to interact
 with  the dataset objects passed as arguments in hte multiview classifiers
 of SuMMIT'''


class Dataset():
    """
    This is the base class for all the type of multiview datasets of SuMMIT.
    """

    @abstractmethod
    def get_nb_samples(self):  # pragma: no cover
        pass

    @abstractmethod
    def get_v(self, view_index, sample_indices=None):  # pragma: no cover
        pass

    @abstractmethod
    def get_label_names(self, sample_indices=None):  # pragma: no cover
        pass

    @abstractmethod
    def get_labels(self, sample_indices=None):  # pragma: no cover
        pass

    @abstractmethod
    def filter(self, labels, label_names, sample_indices, view_names,
               path=None):  # pragma: no cover
        pass

    def init_sample_indices(self, sample_indices=None):
        """
        If no sample indices are provided, selects all the available samples.

        Parameters
        ----------
        sample_indices: np.array,
            An array-like containing the indices of the samples.

        """
        if sample_indices is None:
            return range(self.get_nb_samples())
        else:
            return sample_indices

    def get_shape(self, view_index=0, sample_indices=None):
        """
        Gets the shape of the needed view on the asked samples

        Parameters
        ----------
        view_index : int
            The index of the view to extract
        sample_indices : numpy.ndarray
            The array containing the indices of the samples to extract.

        Returns
        -------
        Tuple containing the shape

        """
        return self.get_v(view_index, sample_indices=sample_indices).shape

    def to_numpy_array(self, sample_indices=None, view_indices=None):
        """
        Concatenates the needed views in one big numpy array while saving the
        limits of each view in a list, to be able to retrieve them later.

        Parameters
        ----------
        sample_indices : array like
            The indices of the samples to extract from the dataset

        view_indices : array like
            The indices of the view to concatenate in the numpy array

        Returns
        -------
        concat_views : numpy array,
            The numpy array containing all the needed views.

        view_limits : list of int
            The limits of each slice used to extract the views.

        """
        view_limits = [0]
        for view_index in view_indices:
            view_data = self.get_v(view_index, sample_indices=sample_indices)
            nb_features = view_data.shape[1]
            view_limits.append(view_limits[-1] + nb_features)
        concat_views = np.concatenate([self.get_v(view_index,
                                                  sample_indices=sample_indices)
                                       for view_index in view_indices], axis=1)
        return concat_views, view_limits

    def select_labels(self, selected_label_names):
        selected_labels = [self.get_label_names().index(label_name.decode())
                           if isinstance(label_name, bytes)
                           else self.get_label_names().index(label_name)
                           for label_name in selected_label_names]
        selected_indices = np.array([index
                                     for index, label in
                                     enumerate(self.get_labels())
                                     if label in selected_labels])
        labels = np.array([selected_labels.index(self.get_labels()[idx])
                           for idx in selected_indices])
        return labels, selected_label_names, selected_indices

    def select_views_and_labels(self, nb_labels=None,
                                selected_label_names=None, random_state=None,
                                view_names=None, path_for_new="../data/"):
        if view_names is None and selected_label_names is None and nb_labels is None:  # pragma: no cover
            pass
        else:
            selected_label_names = self.check_selected_label_names(nb_labels,
                                                                   selected_label_names,
                                                                   random_state)
            labels, label_names, sample_indices = self.select_labels(
                selected_label_names)
            self.filter(labels, label_names, sample_indices, view_names,
                        path_for_new)
        labels_dictionary = dict(
            (labelIndex, labelName) for labelIndex, labelName in
            enumerate(self.get_label_names()))
        return labels_dictionary

    def check_selected_label_names(self, nb_labels=None,
                                   selected_label_names=None,
                                   random_state=np.random.RandomState(42)):
        if selected_label_names is None or nb_labels is None or len(
                selected_label_names) < nb_labels:
            if selected_label_names is None:
                nb_labels_to_add = nb_labels
                selected_label_names = []
            elif nb_labels is not None:
                nb_labels_to_add = nb_labels - len(selected_label_names)
            else:
                nb_labels_to_add = 0
            labels_names_to_choose = [available_label_name
                                      for available_label_name
                                      in self.get_label_names()
                                      if available_label_name
                                      not in selected_label_names]
            added_labels_names = random_state.choice(labels_names_to_choose,
                                                     nb_labels_to_add,
                                                     replace=False)
            selected_label_names = list(selected_label_names) + list(
                added_labels_names)
        elif len(selected_label_names) > nb_labels:
            selected_label_names = list(
                random_state.choice(selected_label_names, nb_labels,
                                    replace=False))

        return selected_label_names

    def gen_feat_id(self):
        self.feature_ids =  [["ID_" + str(i) for i in
                                 range(self.get_v(view_ind).shape[1])]
                                for view_ind in self.view_dict.values()]



class RAMDataset(Dataset):

    def __init__(self, views=None, labels=None, are_sparse=False,
                 view_names=None, labels_names=None, sample_ids=None,
                 name=None, feature_ids=None):
        self.saved_on_disk = False
        self.views = views
        self.labels = np.asarray(labels)
        if isinstance(are_sparse, bool):  # pragma: no cover
            self.are_sparse = [are_sparse for _ in range(len(views))]
        else:
            self.are_sparse = are_sparse
        self.view_names = view_names
        self.labels_names = labels_names
        self.sample_ids = sample_ids
        self.view_dict = dict((view_name, view_ind)
                              for view_name, view_ind
                              in zip(view_names, range(len(views))))
        self.name = name
        self.nb_view = len(self.views)
        self.is_temp = False
        if feature_ids is not None:
            feature_ids = [[feature_id if not is_just_number(feature_id)
                            else "ID_" + feature_id for feature_id in
                            feat_ids] for feat_ids in feature_ids]
            self.feature_ids = feature_ids
        else:
            self.gen_feat_id()

    def get_view_name(self, view_idx):
        return self.view_names[view_idx]

    def init_attrs(self):
        self.nb_view = len(self.views)
        self.view_dict = dict((view_ind, self.view_names[view_ind])
                              for view_ind in range(self.nb_view))

    def get_nb_samples(self):
        return self.views[0].shape[0]

    def get_label_names(self, sample_indices=None, decode=True):
        selected_labels = self.get_labels(sample_indices)
        if decode:
            return [label_name.encode('utf-8')
                    for label, label_name in enumerate(self.labels_names)
                    if label in selected_labels]
        else:
            return [label_name
                    for label, label_name in enumerate(self.labels_names)
                    if label in selected_labels]

    def get_labels(self, sample_indices=None):
        sample_indices = self.init_sample_indices(sample_indices)
        return self.labels[sample_indices]

    def get_v(self, view_index, sample_indices=None):
        sample_indices = self.init_sample_indices(sample_indices)
        if isinstance(sample_indices, int):
            return self.views[view_index][sample_indices, :]
        else:
            sample_indices = np.asarray(sample_indices)
            # sorted_indices = np.argsort(sample_indices)
            # sample_indices = sample_indices[sorted_indices]
            if not self.are_sparse[view_index]:
                return self.views[view_index][
                    sample_indices, :]
            else:  # pragma: no cover
                # TODO Sparse support
                pass

    def get_nb_class(self, sample_indices=None):
        sample_indices = self.init_sample_indices(sample_indices)
        return len(np.unique(self.labels[sample_indices]))

    def filter(self, labels, label_names, sample_indices, view_names,
               path=None):
        if self.sample_ids is not None:
            self.sample_ids = self.sample_ids[sample_indices]
        self.labels = self.labels[sample_indices]
        self.labels_names = [name for lab_index, name
                             in enumerate(self.labels_names)
                             if lab_index in np.unique(self.labels)]
        self.labels = np.array(
            [np.where(label == np.unique(self.labels))[0] for label in
             self.labels])
        self.view_names = view_names
        new_views = []
        for new_view_ind, view_name in enumerate(self.view_names):
            new_views.append(
                self.views[self.view_dict[view_name]][sample_indices, :])
        self.views = new_views
        self.view_dict = dict((view_name, view_ind)
                              for view_ind, view_name
                              in enumerate(self.view_names))
        self.nb_view = len(self.views)

    def get_view_dict(self):
        return self.view_dict

    def get_name(self):
        return self.name


class HDF5Dataset(Dataset):
    """
    Dataset class

    This is used to encapsulate the multiview dataset while keeping it stored on the disk instead of in RAM.


    Parameters
    ----------
    views : list of numpy arrays or None
        The list containing each view of the dataset as a numpy array of shape
        (nb samples, nb features).

    labels : numpy array or None
        The labels for the multiview dataset, of shape (nb samples, ).

    are_sparse : list of bool, or None
        The list of boolean telling if each view is sparse or not.

    file_name : str, or None
        The name of the hdf5 file that will be created to store the multiview
        dataset.

    view_names : list of str, or None
        The name of each view.

    path : str, or None
        The path where the hdf5 dataset file will be stored

    hdf5_file : h5py.File object, or None
        If not None, the dataset will be imported directly from this file.

    labels_names : list of str, or None
        The name for each unique value of the labels given in labels.

    is_temp : bool
        Used if a temporary dataset has to be stored by the benchmark.

    Attributes
    ----------
    dataset : h5py.File object
        The h5py file pbject that points to the hdf5 dataset on the disk.

    nb_view : int
        The number of views in the dataset.

    view_dict : dict
        The dictionnary with the name of each view as the keys and their indices
         as values

    """

    # The following methods use h5py

    def __init__(self, views=None, labels=None, are_sparse=False,
                 file_name="dataset.hdf5", view_names=None, path="",
                 hdf5_file=None, labels_names=None, is_temp=False,
                 sample_ids=None, feature_ids=None):
        self.is_temp = False
        if hdf5_file is not None:
            self.dataset = hdf5_file
            self.init_attrs()
        else:
            secure_file_path(os.path.join(path, file_name))
            dataset_file = h5py.File(os.path.join(path, file_name), "w")
            if view_names is None:
                view_names = ["View" + str(index) for index in
                              range(len(views))]
            if isinstance(are_sparse, bool):  # pragma: no cover
                are_sparse = [are_sparse for _ in views]
            for view_index, (view_name, view, is_sparse) in enumerate(
                    zip(view_names, views, are_sparse)):
                view_dataset = dataset_file.create_dataset(
                    "View" + str(view_index),
                    view.shape,
                    data=view)
                view_dataset.attrs["name"] = view_name
                view_dataset.attrs["sparse"] = is_sparse
            labels_dataset = dataset_file.create_dataset("Labels",
                                                         shape=labels.shape,
                                                         data=labels)
            if labels_names is None:
                labels_names = [str(index) for index in np.unique(labels)]
            labels_dataset.attrs["names"] = [label_name.encode()
                                             if not isinstance(label_name,
                                                               bytes)
                                             else label_name
                                             for label_name in labels_names]
            meta_data_grp = dataset_file.create_group("Metadata")
            meta_data_grp.attrs["nbView"] = len(views)
            meta_data_grp.attrs["nbClass"] = len(np.unique(labels))
            meta_data_grp.attrs["datasetLength"] = len(labels)
            dataset_file.close()
            self.update_hdf5_dataset(os.path.join(path, file_name))
            if sample_ids is not None:
                sample_ids = [sample_id if not is_just_number(sample_id)
                              else "ID_" + sample_id for sample_id in
                              sample_ids]
                self.sample_ids = sample_ids
            else:
                self.sample_ids = ["ID_" + str(i)
                                   for i in range(labels.shape[0])]
            if feature_ids is not None:
                feature_ids = [[feature_id if not is_just_number(feature_id)
                              else "ID_" + feature_id for feature_id in
                              feat_ids] for feat_ids in feature_ids]
                self.feature_ids = feature_ids
            else:
                self.gen_feat_id()


    def get_v(self, view_index, sample_indices=None):
        """ Extract the view and returns a numpy.ndarray containing the description
        of the samples specified in sample_indices

        Parameters
        ----------
        view_index : int
            The index of the view to extract
        sample_indices : numpy.ndarray
            The array containing the indices of the samples to extract.

        Returns
        -------
        A numpy.ndarray containing the view data for the needed samples

        """
        sample_indices = self.init_sample_indices(sample_indices)
        if isinstance(sample_indices, int):
            return self.dataset["View" + str(view_index)][sample_indices, :]
        else:
            sample_indices = np.array(sample_indices)
            # sorted_indices = np.argsort(sample_indices)
            # sample_indices = sample_indices[sorted_indices]

            if not self.dataset["View" + str(view_index)].attrs["sparse"]:
                return self.dataset["View" + str(view_index)][()][
                    sample_indices, :]  # [np.argsort(sorted_indices), :]
            else:  # pragma: no cover
                # Work in progress
                pass

    def get_view_name(self, view_idx):
        """
        Method to get a view's name from its index.

        Parameters
        ----------
        view_idx : int
            The index of the view in the dataset

        Returns
        -------
            The view's name.

        """
        return self.dataset["View" + str(view_idx)].attrs["name"]

    def init_attrs(self):
        """
        Used to init the attributes that are modified when self.dataset
        changes

        Returns
        -------

        """
        self.nb_view = self.dataset["Metadata"].attrs["nbView"]
        self.view_dict = self.get_view_dict()
        self.view_names = [self.dataset["View{}".format(ind)].attrs['name'] for ind in range(self.nb_view)]
        if "sample_ids" in self.dataset["Metadata"].keys():
            self.sample_ids = [sample_id.decode()
                               if not is_just_number(sample_id.decode())
                               else "ID_" + sample_id.decode()
                               for sample_id in
                               self.dataset["Metadata"]["sample_ids"]]
        else:
            self.sample_ids = ["ID_" + str(i) for i in
                               range(self.dataset["Labels"].shape[0])]
        if "feature_ids" in self.dataset["Metadata"].keys():
            self.feature_ids = [[feature_id.decode()
                               if not is_just_number(feature_id.decode())
                               else "ID_" + feature_id.decode()
                               for feature_id in feature_ids] for feature_ids in
                               self.dataset["Metadata"]["feature_ids"]]
        else:
           self.gen_feat_id()

    def get_nb_samples(self):
        """
        Used to get the number of samples available in hte dataset

        Returns
        -------
            int
        """
        return self.dataset["Metadata"].attrs["datasetLength"]

    def get_view_dict(self):
        """
        Returns the dictionary containing view indices as keys and their
        corresponding names as values
        """
        view_dict = {}
        for view_index in range(self.nb_view):
            view_dict[self.dataset["View" + str(view_index)].attrs[
                "name"]] = view_index
        return view_dict

    def get_label_names(self, decode=False, sample_indices=None):
        """
        Used to get the list of the label names for the given set of samples

        Parameters
        ----------
        decode : bool
            If True, will decode the label names before listing them

        sample_indices : numpy.ndarray
            The array containing the indices of the needed samples

        Returns
        -------
            list
            seleted labels' names
        """
        selected_labels = self.get_labels(sample_indices)
        print("selected labels ", selected_labels)
        print("self.dataset ", self.dataset["Labels"].attrs["names"])
        if decode:
            return [label_name.decode("utf-8")
                    for label, label_name in
                    enumerate(self.dataset["Labels"].attrs["names"])
                    if label in selected_labels]
        else:
            return [label_name
                    for label, label_name in
                    enumerate(self.dataset["Labels"].attrs["names"])
                    if label in selected_labels]

    def get_nb_class(self, sample_indices=None):
        """
        Gets the number of classes of the dataset for the asked samples

        Parameters
        ----------
        sample_indices : numpy.ndarray
            The array containing the indices of the samples to extract.

        Returns
        -------
        int : The number of classes

        """
        sample_indices = self.init_sample_indices(sample_indices)
        return len(np.unique(self.dataset["Labels"][()][sample_indices]))

    def get_labels(self, sample_indices=None):
        """Gets the label array for the asked samples

        Parameters
        ----------
        sample_indices : numpy.ndarray
            The array containing the indices of the samples to extract.

        Returns
        -------
        numpy.ndarray containing the labels of the asked samples"""
        sample_indices = self.init_sample_indices(sample_indices)
        return self.dataset["Labels"][()][sample_indices]

    def rm(self):  # pragma: no cover
        """
        Method used to delete the dataset file on the disk if the dataset is
        temporary.

        Returns
        -------

        """
        filename = self.dataset.filename
        self.dataset.close()
        if self.is_temp:
            os.remove(filename)

    def copy_view(self, target_dataset=None, source_view_name=None,
                  target_view_index=None, sample_indices=None):
        sample_indices = self.init_sample_indices(sample_indices)
        new_d_set = target_dataset.create_dataset(
            "View" + str(target_view_index),
            data=self.get_v(self.view_dict[source_view_name],
                            sample_indices=sample_indices))
        for key, value in self.dataset[
                "View" + str(self.view_dict[source_view_name])].attrs.items():
            new_d_set.attrs[key] = value

    def init_view_names(self, view_names=None):
        if view_names is None:
            return [key for key in self.get_view_dict().keys()]
        else:
            return view_names

    def update_hdf5_dataset(self, path):
        if hasattr(self, 'dataset'):
            self.dataset.close()
        self.dataset = h5py.File(path, 'r')
        self.is_temp = True
        self.init_attrs()

    def filter(self, labels, label_names, sample_indices, view_names,
               path=None):
        dataset_file_path = os.path.join(path,
                                         self.get_name() + "_temp_filter.hdf5")
        new_dataset_file = h5py.File(dataset_file_path, "w")
        self.dataset.copy("Metadata", new_dataset_file)
        if "sample_ids" in self.dataset["Metadata"].keys():
            del new_dataset_file["Metadata"]["sample_ids"]
            ex_ids = new_dataset_file["Metadata"].create_dataset("sample_ids",
                                                                 data=np.array(
                                                                     self.sample_ids)[
                                                                     sample_indices].astype(
                                                                     np.dtype(
                                                                         "S100")))
        else:
            new_dataset_file["Metadata"].create_dataset("sample_ids",
                                                        (
                                                            len(
                                                                self.sample_ids),),
                                                        data=np.array(
                                                            self.sample_ids).astype(
                                                            np.dtype("S100")),
                                                        dtype=np.dtype("S100"))
        new_dataset_file["Metadata"].attrs["datasetLength"] = len(
            sample_indices)
        new_dataset_file["Metadata"].attrs["nbClass"] = np.unique(labels)
        new_dataset_file.create_dataset("Labels", data=labels)
        new_dataset_file["Labels"].attrs["names"] = [label_name.encode()
                                                     if not isinstance(
            label_name, bytes)
            else label_name
            for label_name in
            label_names]
        view_names = self.init_view_names(view_names)
        new_dataset_file["Metadata"].attrs["nbView"] = len(view_names)
        for new_index, view_name in enumerate(view_names):
            self.copy_view(target_dataset=new_dataset_file,
                           source_view_name=view_name,
                           target_view_index=new_index,
                           sample_indices=sample_indices)
        new_dataset_file.close()
        self.update_hdf5_dataset(dataset_file_path)

    def add_gaussian_noise(self, random_state, path,
                           noise_std=0.15):
        noisy_dataset = h5py.File(path + self.get_name() + "_noised.hdf5", "w")
        self.dataset.copy("Metadata", noisy_dataset)
        self.dataset.copy("Labels", noisy_dataset)
        for view_index in range(self.nb_view):
            self.copy_view(target_dataset=noisy_dataset,
                           source_view_name=self.get_view_name(view_index),
                           target_view_index=view_index)
        for view_index in range(noisy_dataset["Metadata"].attrs["nbView"]):
            view_key = "View" + str(view_index)
            view_dset = noisy_dataset[view_key]
            view_limits = self.dataset[
                "Metadata/View" + str(view_index) + "_limits"][()]
            view_ranges = view_limits[:, 1] - view_limits[:, 0]
            normal_dist = random_state.normal(
                0, noise_std, view_dset[()].shape)
            noise = normal_dist * view_ranges
            noised_data = view_dset[()] + noise
            noised_data = np.where(noised_data < view_limits[:, 0],
                                   view_limits[:, 0], noised_data)
            noised_data = np.where(noised_data > view_limits[:, 1],
                                   view_limits[:, 1], noised_data)
            noisy_dataset[view_key][...] = noised_data
        noisy_dataset_path = noisy_dataset.filename
        noisy_dataset.close()
        self.update_hdf5_dataset(noisy_dataset_path)

    # The following methods are hdf5 free

    def get_name(self):
        """Gets the name of the dataset hdf5 file"""
        return os.path.split(self.dataset.filename)[-1].split('.')[0]


def is_just_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def datasets_already_exist(pathF, name, nbCores):
    """Used to check if it's necessary to copy datasets"""
    allDatasetExist = True
    for coreIndex in range(nbCores):
        allDatasetExist *= os.path.isfile(os.path.join(
            pathF, name + str(coreIndex) + ".hdf5"))
    return allDatasetExist


def extract_subset(matrix, used_indices):
    """Used to extract a subset of a matrix even if it's sparse WIP"""
    # if sparse.issparse(matrix):
    #     new_indptr = np.zeros(len(used_indices) + 1, dtype=int)
    #     oldindptr = matrix.indptr
    #     for sampleIndexIndex, sampleIndex in enumerate(used_indices):
    #         new_indptr[sampleIndexIndex + 1] = new_indptr[
    #                                                 sampleIndexIndex] + (
    #                                                     oldindptr[
    #                                                         sampleIndex + 1] -
    #                                                     oldindptr[sampleIndex])
    #     new_data = np.ones(new_indptr[-1], dtype=bool)
    #     new_indices = np.zeros(new_indptr[-1], dtype=int)
    #     old_indices = matrix.indices
    #     for sampleIndexIndex, sampleIndex in enumerate(used_indices):
    #         new_indices[new_indptr[sampleIndexIndex]:new_indptr[
    #             sampleIndexIndex + 1]] = old_indices[
    #                                       oldindptr[sampleIndex]:
    #                                       oldindptr[sampleIndex + 1]]
    #     return sparse.csr_matrix((new_data, new_indices, new_indptr),
    #                              shape=(len(used_indices), matrix.shape[1]))
    # else:
    return matrix[used_indices]


def init_multiple_datasets(path_f, name, nb_cores):  # pragma: no cover
    r"""Used to create copies of the dataset if multicore computation is used.

    This is a temporary solution to fix the sharing memory issue with HDF5 datasets.

    Parameters
    ----------
    path_f : string
        Path to the original dataset directory
    name : string
        Name of the dataset
    nb_cores : int
        The number of threads that the benchmark can use

    Returns
    -------
    datasetFiles : None
        Dictionary resuming which mono- and multiview algorithms which will be used in the benchmark.
    """
    if nb_cores > 1:
        if datasets_already_exist(path_f, name, nb_cores):
            logging.info(
                "Info:\t Enough copies of the dataset are already available")
            pass
        else:
            if os.path.getsize(
                    os.path.join(path_f, name + ".hdf5")) * nb_cores / float(
                    1024) / 1000 / 1000 > 0.1:
                logging.info("Start:\t Creating " + str(
                    nb_cores) + " temporary datasets for multiprocessing")
                logging.warning(
                    r" WARNING : /!\ This may use a lot of HDD storage space : " +
                    str(os.path.getsize(os.path.join(path_f,
                                                     name + ".hdf5")) * nb_cores / float(
                        1024) / 1000 / 1000) + " Gbytes /!\\ ")
                confirmation = confirm()
                if not confirmation:
                    sys.exit(0)
                else:
                    pass
            else:
                pass
            dataset_files = copy_hdf5(path_f, name, nb_cores)
            logging.info("Start:\t Creating datasets for multiprocessing")
            return dataset_files


def copy_hdf5(pathF, name, nbCores):
    """Used to copy a HDF5 database in case of multicore computing"""
    datasetFile = h5py.File(pathF + name + ".hdf5", "r")
    for coreIndex in range(nbCores):
        newDataSet = h5py.File(pathF + name + str(coreIndex) + ".hdf5", "w")
        for dataset in datasetFile:
            datasetFile.copy("/" + dataset, newDataSet["/"])
        newDataSet.close()


def delete_HDF5(benchmarkArgumentsDictionaries, nbCores, dataset):
    """Used to delete temporary copies at the end of the benchmark"""
    if nbCores > 1:
        logging.info("Start:\t Deleting " + str(
            nbCores) + " temporary datasets for multiprocessing")
        args = benchmarkArgumentsDictionaries[0]["args"]
        logging.info("Start:\t Deleting datasets for multiprocessing")

        for coreIndex in range(nbCores):
            os.remove(args["pathf"] + args["name"] + str(coreIndex) + ".hdf5")
    if dataset.is_temp:
        dataset.rm()


def confirm(resp=True, timeout=15):  # pragma: no cover
    """Used to process answer"""
    ans = input_(timeout)
    if not ans:
        return resp
    if ans not in ['y', 'Y', 'n', 'N']:
        print('please enter y or n.')
    if ans == 'y' or ans == 'Y':
        return True
    if ans == 'n' or ans == 'N':
        return False


def input_(timeout=15):  # pragma: no cover
    """used as a UI to stop if too much HDD space will be used"""
    logging.warning("You have " + str(
        timeout) + " seconds to stop the dataset copy by typing n")
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        return sys.stdin.readline().strip()
    else:
        return "y"


def get_samples_views_indices(dataset, samples_indices, view_indices, ):
    """This function  is used to get all the samples indices and view indices if needed"""
    if view_indices is None:
        view_indices = np.arange(dataset.nb_view)
    if samples_indices is None:
        samples_indices = np.arange(dataset.get_nb_samples())
    return samples_indices, view_indices
