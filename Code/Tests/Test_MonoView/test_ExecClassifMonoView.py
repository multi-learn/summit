import unittest
import numpy as np
import h5py
import os

from ...MonoMultiViewClassifiers.Monoview import ExecClassifMonoView


class Test_initConstants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.mkdir("Code/Tests/temp_tests")
        cls.datasetFile = h5py.File("Code/Tests/temp_tests/test.hdf5", "w")
        cls.random_state = np.random.RandomState(42)
        cls.args = {"CL_type": "test_clf"}
        cls.X_value = cls.random_state.randint(0,500,(10,20))
        cls.X = cls.datasetFile.create_dataset("View0", data=cls.X_value)
        cls.X.attrs["name"] = "test_dataset"
        cls.X.attrs["sparse"] = False
        cls.classificationIndices = [np.array([0,2,4,6,8]),np.array([1,3,5,7,9])]
        cls.labelsNames = ["test_true", "test_false"]
        cls.name = "test"
        cls.directory = "Code/Tests/temp_tests/test_dir/"

    def test_simple(cls):
        kwargs, \
        t_start, \
        feat, \
        CL_type, \
        X, \
        learningRate, \
        labelsString, \
        outputFileName = ExecClassifMonoView.initConstants(cls.args,
                                                           cls.X,
                                                           cls.classificationIndices,
                                                           cls.labelsNames,
                                                           cls.name,
                                                           cls.directory)
        cls.assertEqual(kwargs, cls.args)
        cls.assertEqual(feat, "test_dataset")
        cls.assertEqual(CL_type, "test_clf")
        np.testing.assert_array_equal(X, cls.X_value)
        cls.assertEqual(learningRate, 0.5)
        cls.assertEqual(labelsString, "test_true-test_false")
        cls.assertEqual(outputFileName, "Code/Tests/temp_tests/test_dir/test_clf/test_dataset/Results-test_clf-test_true-test_false-learnRate0.5-test-test_dataset-")

    @classmethod
    def tearDownClass(cls):
        os.remove("Code/Tests/temp_tests/test.hdf5")
        os.rmdir("Code/Tests/temp_tests/test_dir/test_clf/test_dataset")
        os.rmdir("Code/Tests/temp_tests/test_dir/test_clf")
        os.rmdir("Code/Tests/temp_tests/test_dir")
        os.rmdir("Code/Tests/temp_tests")
