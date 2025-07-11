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
import unittest

import h5py
import numpy as np

from summit.multiview_platform.utils import get_multiview_db
from summit.tests.utils import rm_tmp, tmp_path


class Test_get_classic_db_hdf5(unittest.TestCase):

    def setUp(self):
        rm_tmp()
        os.mkdir(tmp_path)
        self.rs = np.random.RandomState(42)
        self.nb_view = 3
        self.file_name = "test.hdf5"
        self.nb_samples = 5
        self.nb_class = 3
        self.views = [self.rs.randint(0, 10, size=(self.nb_samples, 7))
                      for _ in range(self.nb_view)]
        self.labels = self.rs.randint(0, self.nb_class, self.nb_samples)
        self.dataset_file = h5py.File(
            os.path.join(tmp_path, self.file_name), 'w')
        self.view_names = ["ViewN" + str(index) for index in
                           range(len(self.views))]
        self.are_sparse = [False for _ in self.views]
        for view_index, (view_name, view, is_sparse) in enumerate(
                zip(self.view_names, self.views, self.are_sparse)):
            view_dataset = self.dataset_file.create_dataset(
                "View" + str(view_index),
                view.shape,
                data=view)
            view_dataset.attrs["name"] = view_name
            view_dataset.attrs["sparse"] = is_sparse
        labels_dataset = self.dataset_file.create_dataset("Labels",
                                                          shape=self.labels.shape,
                                                          data=self.labels)
        self.labels_names = [str(index) for index in np.unique(self.labels)]
        labels_dataset.attrs["names"] = [label_name.encode()
                                         for label_name in self.labels_names]
        meta_data_grp = self.dataset_file.create_group("Metadata")
        meta_data_grp.attrs["nbView"] = len(self.views)
        meta_data_grp.attrs["nbClass"] = len(np.unique(self.labels))
        meta_data_grp.attrs["datasetLength"] = len(self.labels)

    def test_simple(self):
        dataset, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(
            ["ViewN2"], tmp_path, self.file_name.split(".")[0],
            self.nb_class, ["0", "2"],
            self.rs, path_for_new=tmp_path)
        self.assertEqual(dataset.nb_view, 1)
        self.assertEqual(labels_dictionary,
                         {0: "0", 1: "2", 2: "1"})
        print(labels_dictionary)
        print(dataset.get_labels())
        self.assertEqual(dataset.get_nb_samples(), 5)
        self.assertEqual(len(np.unique(dataset.get_labels())), 3)

    def test_all_views_asked(self):
        dataset, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(
            None, tmp_path, self.file_name.split(".")[0],
            self.nb_class, ["0", "2"],
            self.rs, path_for_new=tmp_path)
        self.assertEqual(dataset.nb_view, 3)
        self.assertEqual(
            dataset.get_view_dict(), {
                'ViewN0': 0, 'ViewN1': 1, 'ViewN2': 2})

    def test_asked_the_whole_dataset(self):
        dataset, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(
            ["ViewN2"], tmp_path, self.file_name.split(".")[0],
            self.nb_class, ["0", "2"],
            self.rs, path_for_new=tmp_path, full=True)
        self.assertEqual(dataset.dataset, self.dataset_file)

    def tearDown(self):
        self.dataset_file.close()
        rm_tmp()


class Test_get_classic_db_csv(unittest.TestCase):

    def setUp(self):
        rm_tmp()
        os.mkdir(tmp_path)
        self.pathF = tmp_path
        self.NB_CLASS = 2
        self.nameDB = "test_dataset"
        self.askedLabelsNames = ["test_label_1", "test_label_3"]
        self.random_state = np.random.RandomState(42)
        self.views = ["test_view_1", "test_view_3"]
        np.savetxt(self.pathF + self.nameDB + "-labels-names.csv",
                   np.array(["test_label_0", "test_label_1",
                             "test_label_2", "test_label_3"]), fmt="%s",
                   delimiter=",")
        np.savetxt(self.pathF + self.nameDB + "-labels.csv",
                   self.random_state.randint(0, 4, 10), delimiter=",")
        os.mkdir(self.pathF + "Views")
        self.datas = []
        for i in range(4):
            data = self.random_state.randint(0, 100, (10, 20))
            np.savetxt(os.path.join(self.pathF +"Views","test_view_" + str(i) + ".csv"),
                       data, delimiter=",")
            self.datas.append(data)

    def test_simple(self):
        dataset, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_csv(
            self.views, self.pathF, self.nameDB,
            self.NB_CLASS, self.askedLabelsNames,
            self.random_state, delimiter=",", path_for_new=tmp_path)
        self.assertEqual(dataset.nb_view, 2)
        self.assertEqual(
            dataset.get_view_dict(), {
                'test_view_1': 0, 'test_view_3': 1})
        self.assertEqual(labels_dictionary,
                         {0: "test_label_1", 1: "test_label_3"})
        self.assertEqual(dataset.get_nb_samples(), 3)
        self.assertEqual(dataset.get_nb_class(), 2)

    @classmethod
    def tearDown(self):
        rm_tmp()


class Test_get_plausible_db_hdf5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.path = tmp_path
        cls.nb_class = 3
        cls.rs = np.random.RandomState(42)
        cls.nb_view = 3
        cls.nb_samples = 5
        cls.nb_features = 4

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        dataset, labels_dict, name = get_multiview_db.get_plausible_db_hdf5(
            "", self.path, "", nb_class=self.nb_class, random_state=self.rs,
            nb_view=3, nb_samples=self.nb_samples,
            nb_features=self.nb_features)
        self.assertEqual(dataset.init_sample_indices(), range(5))
        self.assertEqual(dataset.get_nb_class(), self.nb_class)

    def test_two_class(self):
        dataset, labels_dict, name = get_multiview_db.get_plausible_db_hdf5(
            "", self.path, "", nb_class=2, random_state=self.rs,
            nb_view=3, nb_samples=self.nb_samples,
            nb_features=self.nb_features)
        self.assertEqual(dataset.init_sample_indices(), range(5))
        self.assertEqual(dataset.get_nb_class(), 2)


if __name__ == '__main__':
    unittest.main()
