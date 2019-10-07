import unittest

import numpy as np
import h5py
import os

from ..utils import rm_tmp, tmp_path

from multiview_platform.mono_multi_view_classifiers.multiview_classifiers import \
    weighted_linear_early_fusion

class Test_WeightedLinearEarlyFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.random_state = np.random.RandomState(42)
        cls.view_weights = [0.5, 0.5]
        os.mkdir("multiview_platform/tests/tmp_tests")
        cls.dataset_file = h5py.File(
            tmp_path+"test_file.hdf5", "w")
        cls.labels = cls.dataset_file.create_dataset("Labels",
                                                     data=np.array([0, 1, 0, 0, 1]))
        cls.view0_data = cls.random_state.randint(1,10,size=(5, 4))
        view0 = cls.dataset_file.create_dataset("View0", data=cls.view0_data)
        view0.attrs["sparse"] = False
        cls.view1_data = cls.random_state.randint(1, 10, size=(5, 4))
        view1 = cls.dataset_file.create_dataset("View1", data=cls.view1_data)
        view1.attrs["sparse"] = False
        metaDataGrp = cls.dataset_file.create_group("Metadata")
        metaDataGrp.attrs["nbView"] = 2
        metaDataGrp.attrs["nbClass"] = 2
        metaDataGrp.attrs["datasetLength"] = 5
        cls.monoview_classifier_name = "decision_tree"
        cls.monoview_classifier_config = {"max_depth":1, "criterion": "gini", "splitter": "best"}
        cls.classifier = weighted_linear_early_fusion.WeightedLinearEarlyFusion(
            random_state=cls.random_state, view_weights=cls.view_weights,
            monoview_classifier_name=cls.monoview_classifier_name,
            monoview_classifier_config=cls.monoview_classifier_config)

    @classmethod
    def tearDownClass(cls):
        cls.dataset_file.close()
        for file_name in os.listdir("multiview_platform/tests/tmp_tests"):
            os.remove(os.path.join("multiview_platform/tests/tmp_tests", file_name))
        os.rmdir("multiview_platform/tests/tmp_tests")

    def test_simple(self):
        np.testing.assert_array_equal(self.view_weights, self.classifier.view_weights)

    def test_fit(self):
        self.assertRaises(AttributeError, getattr,
                          self.classifier.monoview_classifier, "classes_")
        self.classifier.fit(self.dataset_file, self.labels, None, None)
        np.testing.assert_array_equal(self.classifier.monoview_classifier.classes_,
                                      np.array([0,1]))

    def test_predict(self):
        self.classifier.fit(self.dataset_file, self.labels, None, None)
        predicted_labels = self.classifier.predict(self.dataset_file, None, None)
        np.testing.assert_array_equal(predicted_labels, self.labels)

    def test_transform_data_to_monoview_simple(self):


        example_indices, X = self.classifier.transform_data_to_monoview(self.dataset_file,
                                                  None, None)
        self.assertEqual(X.shape, (5,8))
        np.testing.assert_array_equal(X, np.concatenate((self.view0_data, self.view1_data), axis=1))
        np.testing.assert_array_equal(example_indices, np.arange(5))

    def test_transform_data_to_monoview_view_select(self):
        example_indices, X = self.classifier.transform_data_to_monoview(
            self.dataset_file,
            None, np.array([0]))
        self.assertEqual(X.shape, (5, 4))
        np.testing.assert_array_equal(X, self.view0_data)
        np.testing.assert_array_equal(example_indices, np.arange(5))

    def test_transform_data_to_monoview_view_select(self):
        example_indices, X = self.classifier.transform_data_to_monoview(
            self.dataset_file,
            np.array([1,2,3]), np.array([0]))
        self.assertEqual(X.shape, (3, 4))
        np.testing.assert_array_equal(X, self.view0_data[np.array([1,2,3]), :])
        np.testing.assert_array_equal(example_indices, np.array([1,2,3]))

