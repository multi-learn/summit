import unittest
import h5py
import numpy as np
import os

from ...MonoMultiViewClassifiers.utils import GetMultiviewDb

#
# class Test_copyhdf5Dataset(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         if not os.path.exists("Code/Tests/temp_tests"):
#             os.mkdir("Code/Tests/temp_tests")
#         cls.dataset_file = h5py.File("Code/Tests/temp_tests/test_copy.hdf5", "w")
#         cls.dataset = cls.dataset_file.create_dataset("test", data=cls.random_state.randint(0, 100, (10, 20)))
#         cls.dataset.attrs["test_arg"] = "Am I copied"
#
#     def test_simple_copy(cls):
#         GetMultiviewDb.copyhdf5Dataset(cls.dataset_file, cls.dataset_file, "test", "test_copy_1", np.arange(10))
#         np.testing.assert_array_equal(cls.dataset_file.get("test").value, cls.dataset_file.get("test_copy_1").value)
#         cls.assertEqual("Am I copied", cls.dataset_file.get("test_copy_1").attrs["test_arg"])
#
#     def test_copy_only_some_indices(cls):
#         usedIndices = cls.random_state.choice(10,6, replace=False)
#         GetMultiviewDb.copyhdf5Dataset(cls.dataset_file, cls.dataset_file, "test", "test_copy", usedIndices)
#         np.testing.assert_array_equal(cls.dataset_file.get("test").value[usedIndices, :], cls.dataset_file.get("test_copy").value)
#         cls.assertEqual("Am I copied", cls.dataset_file.get("test_copy").attrs["test_arg"])
#
#     @classmethod
#     def tearDownClass(cls):
#         os.remove("Code/Tests/temp_tests/test_copy.hdf5")
#         # for dir in os.listdir("Code/Tests/temp_tests"):print(dir)
#         # import pdb;pdb.set_trace()
#         os.rmdir("Code/Tests/temp_tests")
#
#
#
# class Test_filterViews(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         cls.views = ["test_view_1", "test_view_2"]
#         if not os.path.exists("Code/Tests/temp_tests"):
#             os.mkdir("Code/Tests/temp_tests")
#         cls.dataset_file = h5py.File("Code/Tests/temp_tests/test_copy.hdf5", "w")
#         cls.metadata_group = cls.dataset_file.create_group("Metadata")
#         cls.metadata_group.attrs["nbView"] = 4
#
#         for i in range(4):
#             cls.dataset = cls.dataset_file.create_dataset("View"+str(i),
#                                                           data=cls.random_state.randint(0, 100, (10, 20)))
#             cls.dataset.attrs["name"] = "test_view_"+str(i)
#
#     def test_simple_filter(cls):
#         cls.temp_dataset_file = h5py.File("Code/Tests/temp_tests/test_copy_temp.hdf5", "w")
#         cls.dataset_file.copy("Metadata", cls.temp_dataset_file)
#         GetMultiviewDb.filterViews(cls.dataset_file, cls.temp_dataset_file, cls.views, np.arange(10))
#         cls.assertEqual(cls.dataset_file.get("View1").attrs["name"],
#                         cls.temp_dataset_file.get("View0").attrs["name"])
#         np.testing.assert_array_equal(cls.dataset_file.get("View2").value, cls.temp_dataset_file.get("View1").value)
#         cls.assertEqual(cls.temp_dataset_file.get("Metadata").attrs["nbView"], 2)
#
#     def test_filter_view_and_examples(cls):
#         cls.temp_dataset_file = h5py.File("Code/Tests/temp_tests/test_copy_temp.hdf5", "w")
#         cls.dataset_file.copy("Metadata", cls.temp_dataset_file)
#         usedIndices = cls.random_state.choice(10, 6, replace=False)
#         GetMultiviewDb.filterViews(cls.dataset_file, cls.temp_dataset_file, cls.views, usedIndices)
#         np.testing.assert_array_equal(cls.dataset_file.get("View1").value[usedIndices, :],
#                                       cls.temp_dataset_file.get("View0").value)
#
#     @classmethod
#     def tearDownClass(cls):
#         os.remove("Code/Tests/temp_tests/test_copy.hdf5")
#         os.remove("Code/Tests/temp_tests/test_copy_temp.hdf5")
#         os.rmdir("Code/Tests/temp_tests")
#
# #
# class Test_filterLabels(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         cls.labelsSet = set(range(4))
#         cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
#         cls.fullLabels = cls.random_state.randint(0,4,10)
#         cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#         cls.askedLabelsNames = ["test_label_1", "test_label_3"]
#
#     def test_simple(cls):
#         newLabels, \
#         newLabelsNames, \
#         usedIndices = GetMultiviewDb.filterLabels(cls.labelsSet,
#                                                   cls.askedLabelsNamesSet,
#                                                   cls.fullLabels,
#                                                   cls.availableLabelsNames,
#                                                   cls.askedLabelsNames)
#         cls.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
#         np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
#         np.testing.assert_array_equal(newLabels, np.array([1,1,0]))
#
#     def test_biclasse(cls):
#         cls.labelsSet = {0,1}
#         cls.fullLabels = cls.random_state.randint(0,2,10)
#         cls.availableLabelsNames = ["test_label_0", "test_label_1"]
#         newLabels, \
#         newLabelsNames, \
#         usedIndices = GetMultiviewDb.filterLabels(cls.labelsSet,
#                                                   cls.askedLabelsNamesSet,
#                                                   cls.fullLabels,
#                                                   cls.availableLabelsNames,
#                                                   cls.askedLabelsNames)
#         cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
#         np.testing.assert_array_equal(usedIndices, np.arange(10))
#         np.testing.assert_array_equal(newLabels, cls.fullLabels)
#
#     def test_asked_too_many_labels(cls):
#         cls.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3", "chicken_is_heaven"}
#         with cls.assertRaises(GetMultiviewDb.DatasetError) as catcher:
#             GetMultiviewDb.filterLabels(cls.labelsSet,
#                                         cls.askedLabelsNamesSet,
#                                         cls.fullLabels,
#                                         cls.availableLabelsNames,
#                                         cls.askedLabelsNames)
#         exception = catcher.exception
#         # cls.assertTrue("Asked more labels than available in the dataset. Available labels are : test_label_0, test_label_1, test_label_2, test_label_3" in exception)
#     #
#     def test_asked_all_labels(cls):
#         cls.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3"}
#         cls.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#         newLabels, \
#         newLabelsNames, \
#         usedIndices = GetMultiviewDb.filterLabels(cls.labelsSet,
#                                                   cls.askedLabelsNamesSet,
#                                                   cls.fullLabels,
#                                                   cls.availableLabelsNames,
#                                                   cls.askedLabelsNames)
#         cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
#         np.testing.assert_array_equal(usedIndices, np.arange(10))
#         np.testing.assert_array_equal(newLabels, cls.fullLabels)
#
#
# class Test_selectAskedLabels(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
#         cls.fullLabels = cls.random_state.randint(0, 4, 10)
#         cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#         cls.askedLabelsNames = ["test_label_1", "test_label_3"]
#
#     def test_simple(cls):
#         newLabels, \
#         newLabelsNames, \
#         usedIndices = GetMultiviewDb.selectAskedLabels(cls.askedLabelsNamesSet,
#                                                        cls.availableLabelsNames,
#                                                        cls.askedLabelsNames,
#                                                        cls.fullLabels)
#         cls.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
#         np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
#         np.testing.assert_array_equal(newLabels, np.array([1, 1, 0]))
#
#     def test_asked_all_labels(cls):
#         cls.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3"}
#         cls.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#         newLabels, \
#         newLabelsNames, \
#         usedIndices = GetMultiviewDb.selectAskedLabels(cls.askedLabelsNamesSet,
#                                                        cls.availableLabelsNames,
#                                                        cls.askedLabelsNames,
#                                                        cls.fullLabels)
#         cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
#         np.testing.assert_array_equal(usedIndices, np.arange(10))
#         np.testing.assert_array_equal(newLabels, cls.fullLabels)
#
#     def test_asked_unavailable_labels(cls):
#         cls.askedLabelsNamesSet = {"test_label_1", "test_label_3", "chicken_is_heaven"}
#         with cls.assertRaises(GetMultiviewDb.DatasetError) as catcher:
#             GetMultiviewDb.selectAskedLabels(cls.askedLabelsNamesSet,
#                                              cls.availableLabelsNames,
#                                              cls.askedLabelsNames,
#                                              cls.fullLabels)
#         exception = catcher.exception
#         # cls.assertTrue("Asked labels are not all available in the dataset" in exception)
#
#
# class Test_getAllLabels(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#         cls.fullLabels = cls.random_state.randint(0, 4, 10)
#         cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#
#     def test_simple(cls):
#         newLabels, newLabelsNames, usedIndices = GetMultiviewDb.getAllLabels(cls.fullLabels, cls.availableLabelsNames)
#         cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
#         np.testing.assert_array_equal(usedIndices, np.arange(10))
#         np.testing.assert_array_equal(newLabels, cls.fullLabels)
#
#
# class Test_fillLabelNames(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.NB_CLASS = 2
#         cls.askedLabelsNames = ["test_label_1", "test_label_3"]
#         cls.randomState = np.random.RandomState(42)
#         cls.availableLabelsNames = ["test_label_"+str(_) for _ in range(40)]
#
#     def test_simple(cls):
#         askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(cls.NB_CLASS,
#                                                                               cls.askedLabelsNames,
#                                                                               cls.randomState,
#                                                                               cls.availableLabelsNames)
#         cls.assertEqual(askedLabelsNames, cls.askedLabelsNames)
#         cls.assertEqual(askedLabelsNamesSet, set(cls.askedLabelsNames))
#
#     def test_missing_labels_names(cls):
#         cls.NB_CLASS = 39
#         askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(cls.NB_CLASS,
#                                                                               cls.askedLabelsNames,
#                                                                               cls.randomState,
#                                                                               cls.availableLabelsNames)
#
#         cls.assertEqual(askedLabelsNames, ['test_label_1', 'test_label_3', 'test_label_35', 'test_label_38', 'test_label_6', 'test_label_15', 'test_label_32', 'test_label_28', 'test_label_8', 'test_label_29', 'test_label_26', 'test_label_17', 'test_label_19', 'test_label_10', 'test_label_18', 'test_label_14', 'test_label_21', 'test_label_11', 'test_label_34', 'test_label_0', 'test_label_27', 'test_label_7', 'test_label_13', 'test_label_2', 'test_label_39', 'test_label_23', 'test_label_4', 'test_label_31', 'test_label_37', 'test_label_5', 'test_label_36', 'test_label_25', 'test_label_33', 'test_label_12', 'test_label_24', 'test_label_20', 'test_label_22', 'test_label_9', 'test_label_16'])
#         cls.assertEqual(askedLabelsNamesSet, set(["test_label_"+str(_) for _ in range(30)]+["test_label_"+str(31+_) for _ in range(9)]))
#
#     def test_too_many_label_names(cls):
#         cls.NB_CLASS = 2
#         cls.askedLabelsNames = ["test_label_1", "test_label_3", "test_label_4", "test_label_6"]
#         askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(cls.NB_CLASS,
#                                                                               cls.askedLabelsNames,
#                                                                               cls.randomState,
#                                                                               cls.availableLabelsNames)
#         cls.assertEqual(askedLabelsNames, ["test_label_3", "test_label_6"])
#         cls.assertEqual(askedLabelsNamesSet, {"test_label_3", "test_label_6"})
#
#
# class Test_allAskedLabelsAreAvailable(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
#         cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#
#     def test_asked_available_labels(cls):
#         cls.assertTrue(GetMultiviewDb.allAskedLabelsAreAvailable(cls.askedLabelsNamesSet,cls.availableLabelsNames))
#
#     def test_asked_unavailable_label(cls):
#         cls.askedLabelsNamesSet = {"test_label_1", "test_label_3", "chicken_is_heaven"}
#         cls.assertFalse(GetMultiviewDb.allAskedLabelsAreAvailable(cls.askedLabelsNamesSet,cls.availableLabelsNames))
#
#
# class Test_getClasses(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.random_state = np.random.RandomState(42)
#
#     def test_multiclass(cls):
#         labelsSet = GetMultiviewDb.getClasses(cls.random_state.randint(0,5,30))
#         cls.assertEqual(labelsSet, {0,1,2,3,4})
#
#     def test_biclass(cls):
#         labelsSet = GetMultiviewDb.getClasses(cls.random_state.randint(0,2,30))
#         cls.assertEqual(labelsSet, {0,1})
#
#     def test_one_class(cls):
#         with cls.assertRaises(GetMultiviewDb.DatasetError) as catcher:
#             GetMultiviewDb.getClasses(np.zeros(30,dtype=int))
#         exception = catcher.exception
#         # cls.assertTrue("Dataset must have at least two different labels" in exception)
#
#
# class Test_getClassicDBhdf5(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         if not os.path.exists("Code/Tests/temp_tests"):
#             os.mkdir("Code/Tests/temp_tests")
#         cls.dataset_file = h5py.File("Code/Tests/temp_tests/test_dataset.hdf5", "w")
#         cls.pathF = "Code/Tests/temp_tests/"
#         cls.nameDB = "test_dataset"
#         cls.NB_CLASS = 2
#         cls.askedLabelsNames = ["test_label_1", "test_label_3"]
#         cls.random_state = np.random.RandomState(42)
#         cls.views = ["test_view_1", "test_view_3"]
#         cls.metadata_group = cls.dataset_file.create_group("Metadata")
#         cls.metadata_group.attrs["nbView"] = 4
#         cls.labels_dataset = cls.dataset_file.create_dataset("Labels", data=cls.random_state.randint(0,4,10))
#         cls.labels_dataset.attrs["names"] = ["test_label_0".encode(), "test_label_1".encode(), "test_label_2".encode(), "test_label_3".encode()]
#
#         for i in range(4):
#             cls.dataset = cls.dataset_file.create_dataset("View" + str(i),
#                                                           data=cls.random_state.randint(0, 100, (10, 20)))
#             cls.dataset.attrs["name"] = "test_view_" + str(i)
#
#     def test_simple(cls):
#         dataset_file, labels_dictionary = GetMultiviewDb.getClassicDBhdf5(cls.views, cls.pathF, cls.nameDB,
#                                                                            cls.NB_CLASS, cls.askedLabelsNames,
#                                                                            cls.random_state)
#         cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
#         cls.assertEqual(labels_dictionary, {0:"test_label_1", 1:"test_label_3"})
#         cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)
#         np.testing.assert_array_equal(dataset_file.get("View0").value, cls.dataset_file.get("View1").value[np.array([1,5,9]),:])
#
#     def test_all_labels_asked(cls):
#         askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#         NB_CLASS = 4
#         dataset_file, labels_dictionary = GetMultiviewDb.getClassicDBhdf5(cls.views, cls.pathF, cls.nameDB,
#                                                                            NB_CLASS, askedLabelsNames,
#                                                                            cls.random_state)
#         cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
#         cls.assertEqual(labels_dictionary, {0:"test_label_0", 1:"test_label_1", 2:"test_label_2", 3:"test_label_3"})
#         cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 10)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 4)
#         np.testing.assert_array_equal(dataset_file.get("View0").value, cls.dataset_file.get("View1").value)
#
#     def test_all_views_asked(cls):
#         views = ["test_view_0", "test_view_1", "test_view_2", "test_view_3"]
#         dataset_file, labels_dictionary = GetMultiviewDb.getClassicDBhdf5(views, cls.pathF, cls.nameDB,
#                                                                            cls.NB_CLASS, cls.askedLabelsNames,
#                                                                            cls.random_state)
#         for viewIndex in range(4):
#             np.testing.assert_array_equal(dataset_file.get("View"+str(viewIndex)).value, cls.dataset_file.get("View"+str(viewIndex)).value[np.array([1,5,9]),:])
#             cls.assertEqual(dataset_file.get("View"+str(viewIndex)).attrs["name"], "test_view_"+str(viewIndex))
#         cls.assertEqual(labels_dictionary, {0:"test_label_1", 1:"test_label_3"})
#         cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 4)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)
#
#     def test_asked_the_whole_dataset(cls):
#         askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
#         NB_CLASS = 4
#         views = ["test_view_0", "test_view_1", "test_view_2", "test_view_3"]
#         dataset_file, labels_dictionary = GetMultiviewDb.getClassicDBhdf5(views, cls.pathF, cls.nameDB,
#                                                                            NB_CLASS, askedLabelsNames,
#                                                                            cls.random_state)
#         for viewIndex in range(4):
#             np.testing.assert_array_equal(dataset_file.get("View"+str(viewIndex)).value, cls.dataset_file.get("View"+str(viewIndex)))
#             cls.assertEqual(dataset_file.get("View"+str(viewIndex)).attrs["name"], "test_view_"+str(viewIndex))
#         cls.assertEqual(labels_dictionary, {0:"test_label_0", 1:"test_label_1", 2:"test_label_2", 3:"test_label_3"})
#         cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 10)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 4)
#         cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 4)
#
#     @classmethod
#     def tearDownClass(cls):
#         os.remove("Code/Tests/temp_tests/test_dataset_temp_view_label_select.hdf5")
#         os.remove("Code/Tests/temp_tests/test_dataset.hdf5")
#         dirs = os.listdir("Code/Tests/temp_tests")
#         for dir in dirs:
#             print(dir)
#         os.rmdir("Code/Tests/temp_tests")

class Test_getClassicDBcsv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        cls.pathF = "Code/Tests/temp_tests/"
        cls.NB_CLASS = 2
        cls.nameDB = "test_dataset"
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]
        cls.random_state = np.random.RandomState(42)
        cls.views = ["test_view_1", "test_view_3"]
        np.savetxt(cls.pathF+cls.nameDB+"-labels-names.csv", np.array(["test_label_0", "test_label_1",
                                                              "test_label_2", "test_label_3"]), fmt="%s", delimiter=",")
        np.savetxt(cls.pathF+cls.nameDB+"-labels.csv", cls.random_state.randint(0,4,10), delimiter=",")
        os.mkdir(cls.pathF+"Views")
        for i in range(4):
            if i == 1:
                print(cls.random_state.randint(0, 100, (10, 20))[np.array([1,5,9]),:])
            np.savetxt(cls.pathF+"Views/test_view_" + str(i)+".csv",
                       cls.random_state.randint(0, 100, (10, 20)), delimiter=",")


    def test_simple(cls):
        dataset_file, labels_dictionary = GetMultiviewDb.getClassicDBcsv(cls.views, cls.pathF, cls.nameDB,
                                                                       cls.NB_CLASS, cls.askedLabelsNames,
                                                                       cls.random_state, delimiter=",")
        # for key in dataset_file.keys():print(key)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
        cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
        cls.assertEqual(labels_dictionary, {0:"test_label_1", 1:"test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)
        # cls.expected_value_0 = np.array([np.array([57, 21, 88, 48, 90, 58, 41, 91, 59, 79, 14, 61, 61, 46, 61, 50, 54, 63, 2, 50]),
        #                                np.array([44, 64, 88, 70, 8, 87, 0, 7, 87, 62, 10, 80, 7, 34, 34, 32, 4, 40, 27, 6]),
        #                                np.array([61, 74, 91, 88, 61, 96, 0, 26, 61, 76, 2, 69, 71, 26, 8, 61, 36, 96, 50, 43])])
        # cls.expected_value_1 = np.array([np.array([15, 13, 75, 86, 14, 91, 97, 65, 31, 86, 62, 85, 50, 24, 57, 62, 61, 21, 57, 57]),
        #                                np.array([28, 77, 91, 68, 46, 93, 61, 68, 75, 15, 89, 89, 47, 84, 38, 99, 32, 93, 22, 9]),
        #                                np.array([20, 35, 9, 72, 23, 63, 98, 48, 98, 35, 81, 95, 23, 22, 61, 95, 36, 11, 54, 12])])
        # cls.expected_value_2 = np.array([np.array([72, 0, 50, 44, 76, 3, 61, 64, 31, 33, 91, 94, 71, 38, 25, 33, 53, 2, 49, 11]),
        #                                np.array([7, 3, 3, 55, 24, 66, 95, 66, 26, 92, 31, 49, 60, 50, 18, 20, 4, 81, 91, 41]),
        #                                np.array([85, 89, 43, 24, 16, 12, 83, 24, 67, 9, 66, 17, 99, 85, 33, 7, 39, 82, 41, 40])])
        # cls.expected_value_3 = np.array([np.array([12, 61, 81, 88, 96, 59, 42, 75, 99, 67, 4, 36, 71, 91, 30, 8, 50, 28, 77, 39]),
        #                                np.array([70, 80, 83, 48, 19, 85, 91, 62, 60, 48, 70, 0, 95, 12, 93, 86, 50, 55, 82, 61]),
        #                                np.array([15, 67, 36, 53, 84, 13, 94, 54, 47, 81, 6, 73, 6, 32, 22, 84, 18, 18, 35, 28])])
        cls.expected_values = [cls.expected_value_0, cls.expected_value_1, cls.expected_value_2, cls.expected_value_3]
        np.testing.assert_array_equal(dataset_file.get("View0").value, cls.expected_value_1)
        np.testing.assert_array_equal(dataset_file.get("View1").value, cls.expected_value_3)

    @classmethod
    def tearDownClass(cls):
        for i in range(4):
            os.remove("Code/Tests/temp_tests/Views/test_view_"+str(i)+".csv")
        os.rmdir("Code/Tests/temp_tests/Views")
        os.remove("Code/Tests/temp_tests/test_dataset-labels-names.csv")
        os.remove("Code/Tests/temp_tests/test_dataset-labels.csv")
        os.remove("Code/Tests/temp_tests/test_dataset.hdf5")
        os.remove("Code/Tests/temp_tests/test_dataset_temp_view_label_select.hdf5")
        for file in os.listdir("Code/Tests/temp_tests"):print(file)
        os.rmdir("Code/Tests/temp_tests")