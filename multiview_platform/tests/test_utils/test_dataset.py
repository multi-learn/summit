import unittest
import h5py
import numpy as np
import os

from ..utils import rm_tmp, tmp_path

from ...mono_multi_view_classifiers.utils import dataset


class Test_Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        cls.rs = np.random.RandomState(42)
        cls.nb_view = 3
        cls.file_name = "test.hdf5"
        cls.nb_examples = 5
        cls.nb_class = 3
        cls.views = [cls.rs.randint(0,10,size=(cls.nb_examples,7))
                 for _ in range(cls.nb_view)]
        cls.labels = cls.rs.randint(0,cls.nb_class,cls.nb_examples)
        cls.dataset_file = h5py.File(os.path.join(tmp_path, cls.file_name))
        cls.view_names = ["ViewN" + str(index) for index in range(len(cls.views))]
        cls.are_sparse = [False for _ in cls.views]
        for view_index, (view_name, view, is_sparse) in enumerate(
                zip(cls.view_names, cls.views, cls.are_sparse)):
            view_dataset = cls.dataset_file.create_dataset("View" + str(view_index),
                                                       view.shape,
                                                       data=view)
            view_dataset.attrs["name"] = view_name
            view_dataset.attrs["sparse"] = is_sparse
        labels_dataset = cls.dataset_file.create_dataset("Labels",
                                                     shape=cls.labels.shape,
                                                     data=cls.labels)
        cls.labels_names = [str(index) for index in np.unique(cls.labels)]
        labels_dataset.attrs["names"] = [label_name.encode()
                                         for label_name in cls.labels_names]
        meta_data_grp = cls.dataset_file.create_group("Metadata")
        meta_data_grp.attrs["nbView"] = len(cls.views)
        meta_data_grp.attrs["nbClass"] = len(np.unique(cls.labels))
        meta_data_grp.attrs["datasetLength"] = len(cls.labels)

    @classmethod
    def tearDownClass(cls):
        cls.dataset_file.close()
        rm_tmp()

    def test_simple(self):
        dataset_object = dataset.Dataset(hdf5_file=self.dataset_file)

    def test_init_example_indices(self):
        example_indices = dataset.Dataset(hdf5_file=self.dataset_file).init_example_indces()
        self.assertEqual(example_indices, range(self.nb_examples))
        example_indices = dataset.Dataset(hdf5_file=self.dataset_file).init_example_indces([0,1,2])
        self.assertEqual(example_indices, [0,1,2])

    def test_get_v(self):
        view = dataset.Dataset(hdf5_file=self.dataset_file).get_v(0)
        np.testing.assert_array_equal(view, self.views[0])
        view = dataset.Dataset(hdf5_file=self.dataset_file).get_v(1, [0,1,2])
        np.testing.assert_array_equal(view, self.views[1][[0,1,2,], :])

    def test_get_nb_class(self):
        nb_class = dataset.Dataset(hdf5_file=self.dataset_file).get_nb_class()
        self.assertEqual(nb_class, self.nb_class)
        nb_class = dataset.Dataset(hdf5_file=self.dataset_file).get_nb_class([0])
        self.assertEqual(nb_class, 1)

    def test_from_scratch(self):
        dataset_object = dataset.Dataset(views=self.views,
                                             labels=self.labels,
                                             are_sparse=self.are_sparse,
                                             file_name="from_scratch"+self.file_name,
                                             view_names=self.view_names,
                                             path=tmp_path,
                                             labels_names=self.labels_names)
        nb_class = dataset_object.get_nb_class()
        self.assertEqual(nb_class, self.nb_class)
        example_indices = dataset_object.init_example_indces()
        self.assertEqual(example_indices, range(self.nb_examples))
        view = dataset_object.get_v(0)
        np.testing.assert_array_equal(view, self.views[0])

    def test_get_view_dict(self):
        dataset_object = dataset.Dataset(views=self.views,
                                         labels=self.labels,
                                         are_sparse=self.are_sparse,
                                         file_name="from_scratch" + self.file_name,
                                         view_names=self.view_names,
                                         path=tmp_path,
                                         labels_names=self.labels_names)
        self.assertEqual(dataset_object.get_view_dict(), {"ViewN0":0,
                                                          "ViewN1": 1,
                                                          "ViewN2": 2,})