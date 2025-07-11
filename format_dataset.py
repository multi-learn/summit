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
"""
This file is provided as an example of dataset formatting, using a csv-stored
mutliview dataset to build a SuMMIT-compatible hdf5 file.
Please see http://baptiste.bauvin.pages.lis-lab.fr/multiview-machine-learning-omis/tutorials/example4.html
for complementary information the example given here is fully described in the
documentation.
"""

import numpy as np
import h5py


# The following variables are defined as an example, you should modify them to fit your dataset files.
view_names = ["sound", "image", "commentary", ]
data_file_paths = ["path/to/sound.csv", "path/to/image.csv", "path/to/commentary.csv",]
labels_file_path = "path/to/labels/file.csv"
sample_ids_path = "path/to/sample_ids/file.csv"
labels_names = ["Human", "Animal", "Object"]


# HDF5 dataset initialization :
hdf5_file = h5py.File("path/to/file.hdf5", "w")

# Store each view in a hdf5 dataset :
for view_index, (file_path, view_name) in enumerate(zip(data_file_paths, view_names)):
    # Get the view's data from the csv file
    view_data = np.genfromtxt(file_path, delimiter=",")

    # Store it in a dataset in the hdf5 file,
    # do not modify the name of the dataset
    view_dataset = hdf5_file.create_dataset(name="View{}".format(view_index),
                                            shape=view_data.shape,
                                            data=view_data)
    # Store the name of the view in an attribute,
    # do not modify the attribute's key
    view_dataset.attrs["name"] = view_name

    # This is an artifact of work in progress for sparse support, not available ATM,
    # do not modify the attribute's key
    view_dataset.attrs["sparse"] = False

# Get le labels data from a csv file
labels_data = np.genfromtxt(labels_file_path, delimiter=',')

# Here, we supposed that the labels file contained numerical labels (0,1,2)
# that refer to the label names of label_names.
# The Labels HDF5 dataset must contain only integers that represent the
# different classes, the names of each class are saved in an attribute

# Store the integer labels in the HDF5 dataset,
# do not modify the name of the dataset
labels_dset = hdf5_file.create_dataset(name="Labels",
                                       shape=labels_data.shape,
                                       data=labels_data)

# Save the labels names in an attribute as encoded strings,
# do not modify the attribute's key
labels_dset.attrs["names"] = [label_name.encode() for label_name in labels_names]

# Create a Metadata HDF5 group to store the metadata,
# do not modify the name of the group
metadata_group = hdf5_file.create_group(name="Metadata")

# Store the number of views in the dataset,
# do not modify the attribute's key
metadata_group.attrs["nbView"] = len(view_names)

# Store the number of classes in the dataset,
# do not modify the attribute's key
metadata_group.attrs["nbClass"] = np.unique(labels_data)

# Store the number of samples in the dataset,
# do not modify the attribute's key
metadata_group.attrs["datasetLength"] = labels_data.shape[0]

# Let us suppose that the samples have string ids, available in a csv file,
# they can be stored in the HDF5 and will be used in the result analysis.
sample_ids = np.genfromtxt(sample_ids_path, delimiter=',')

# To sore the strings in an HDF5 dataset, be sure to use the S<max_length> type,
# do not modify the name of the dataset.
metadata_group.create_dataset("sample_ids",
                              data=np.array(sample_ids).astype(np.dtype("S100")),
                              dtype=np.dtype("S100"))

hdf5_file.close()