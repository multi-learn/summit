import logging
import os
import select
import sys

import h5py
import numpy as np
from scipy import sparse

from . import get_multiview_db as DB


def get_v(dataset, view_index, used_indices=None):
    """Used to extract a view as a numpy array or a sparse mat from the HDF5 dataset"""
    if used_indices is None:
        used_indices = range(dataset.get("Metadata").attrs["datasetLength"])
    if type(used_indices) is int:
        return dataset.get("View" + str(view_index))[used_indices, :]
    else:
        used_indices = np.array(used_indices)
        sorted_indices = np.argsort(used_indices)
        used_indices = used_indices[sorted_indices]

        if not dataset.get("View" + str(view_index)).attrs["sparse"]:
            return dataset.get("View" + str(view_index))[used_indices, :][
                   np.argsort(sorted_indices), :]
        else:
            sparse_mat = sparse.csr_matrix(
                (dataset.get("View" + str(view_index)).get("data").value,
                 dataset.get("View" + str(view_index)).get("indices").value,
                 dataset.get("View" + str(view_index)).get("indptr").value),
                shape=dataset.get("View" + str(view_index)).attrs["shape"])[
                         used_indices, :][
                         np.argsort(sorted_indices), :]

            return sparse_mat


def get_shape(dataset, view_index):
    """Used to get the dataset shape even if it's sparse"""
    if not dataset.get("View" + str(view_index)).attrs["sparse"]:
        return dataset.get("View" + str(view_index)).shape
    else:
        return dataset.get("View" + str(view_index)).attrs["shape"]


def get_value(dataset):
    """Used to get the value of a view in the HDF5 dataset even if it sparse"""
    if not dataset.attrs["sparse"]:
        return dataset.value
    else:
        sparse_mat = sparse.csr_matrix((dataset.get("data").value,
                                        dataset.get("indices").value,
                                        dataset.get("indptr").value),
                                       shape=dataset.attrs["shape"])
        return sparse_mat


def extract_subset(matrix, used_indices):
    """Used to extract a subset of a matrix even if it's sparse"""
    if sparse.issparse(matrix):
        new_indptr = np.zeros(len(used_indices) + 1, dtype=int)
        oldindptr = matrix.indptr
        for exampleIndexIndex, exampleIndex in enumerate(used_indices):
            new_indptr[exampleIndexIndex + 1] = new_indptr[exampleIndexIndex] + (
                    oldindptr[exampleIndex + 1] - oldindptr[exampleIndex])
        new_data = np.ones(new_indptr[-1], dtype=bool)
        new_indices = np.zeros(new_indptr[-1], dtype=int)
        old_indices = matrix.indices
        for exampleIndexIndex, exampleIndex in enumerate(used_indices):
            new_indices[new_indptr[exampleIndexIndex]:new_indptr[
                exampleIndexIndex + 1]] = old_indices[
                                          oldindptr[exampleIndex]:
                                          oldindptr[exampleIndex + 1]]
        return sparse.csr_matrix((new_data, new_indices, new_indptr),
                                 shape=(len(used_indices), matrix.shape[1]))
    else:
        return matrix[used_indices]


def init_multiple_datasets(path_f, name, nb_cores):
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
        if DB.datasetsAlreadyExist(path_f, name, nb_cores):
            logging.debug(
                "Info:\t Enough copies of the dataset are already available")
            pass
        else:
            logging.debug("Start:\t Creating " + str(
                nb_cores) + " temporary datasets for multiprocessing")
            logging.warning(
                " WARNING : /!\ This may use a lot of HDD storage space : " +
                str(os.path.getsize(path_f + name + ".hdf5") * nb_cores / float(
                    1024) / 1000 / 1000) + " Gbytes /!\ ")
            confirmation = confirm()
            if not confirmation:
                sys.exit(0)
            else:
                dataset_files = DB.copyHDF5(path_f, name, nb_cores)
                logging.debug("Start:\t Creating datasets for multiprocessing")
                return dataset_files


def confirm(resp=True, timeout=15):
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


def input_(timeout=15):
    """used as a UI to stop if too much HDD space will be used"""
    logging.warning("You have " + str(
        timeout) + " seconds to stop the dataset copy by typing n")
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        return sys.stdin.readline().strip()
    else:
        return "y"

def get_monoview_shared(path, name, view_name, labels_names, classification_indices):
    """ATM is not used with shared memory, but soon :)"""
    hdf5_dataset_file = h5py.File(path + name + ".hdf5", "w")
    X = hdf5_dataset_file.get(view_name).value
    y = hdf5_dataset_file.get("Labels").value
    return X, y
