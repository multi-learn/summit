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
import numpy as np
import h5py

from ..multiview_platform.utils.dataset import HDF5Dataset


tmp_path = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "tmp_tests", "")
# TODO Convert to ram dataset
test_dataset = HDF5Dataset(
    hdf5_file=h5py.File(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "test_database.hdf5"),
        "r"))


def rm_tmp(path=tmp_path):
    try:
        for file_name in os.listdir(path):
            if os.path.isdir(os.path.join(path, file_name)):
                rm_tmp(os.path.join(path, file_name))
            else:
                os.remove(os.path.join(path, file_name))
        os.rmdir(path)
    except BaseException:
        pass


def gen_test_dataset(random_state=np.random.RandomState(42)):
    dataset_file = h5py.File("test_database.hdf5", "w")
    view_names = ["ViewN0", "ViewN1", "ViewN2"]
    views = [random_state.randint(0, 100, (5, 6))
             for _ in range(len(view_names))]
    labels = random_state.randint(0, 2, 5)
    label_names = ["yes", "no"]
    for view_index, (view_name, view) in enumerate(
            zip(view_names, views)):
        view_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                   view.shape,
                                                   data=view)
        view_dataset.attrs["name"] = view_name
        view_dataset.attrs["sparse"] = False
    labels_dataset = dataset_file.create_dataset("Labels",
                                                 shape=labels.shape,
                                                 data=labels)
    labels_dataset.attrs["names"] = [label_name.encode()
                                     if not isinstance(label_name, bytes)
                                     else label_name
                                     for label_name in label_names]
    meta_data_grp = dataset_file.create_group("Metadata")
    meta_data_grp.attrs["nbView"] = len(views)
    meta_data_grp.attrs["nbClass"] = len(np.unique(labels))
    meta_data_grp.attrs["datasetLength"] = len(labels)
    dataset_file.close()


if __name__ == "__main__":
    gen_test_dataset()
