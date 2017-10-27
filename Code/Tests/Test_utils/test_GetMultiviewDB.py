import unittest
import h5py
import numpy as np
import os

from ...MonoMultiViewClassifiers.utils import GetMultiviewDb


class Test_copyhdf5Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        cls.dataset_file = h5py.File("Code/Tests/temp_tests/test_copy.hdf5", "w")
        cls.dataset = cls.dataset_file.create_dataset("test", data=cls.random_state.randint(0, 100, (10, 20)))
        cls.dataset.attrs["test_arg"] = "Am I copied"

    def test_simple_copy(cls):
        GetMultiviewDb.copyhdf5Dataset(cls.dataset_file, cls.dataset_file, "test", "test_copy_1", np.arange(10))
        np.testing.assert_array_equal(cls.dataset_file.get("test").value, cls.dataset_file.get("test_copy_1").value)
        cls.assertEqual("Am I copied", cls.dataset_file.get("test_copy_1").attrs["test_arg"])

    def test_copy_only_some_indices(cls):
        usedIndices = cls.random_state.choice(10,6, replace=False)
        GetMultiviewDb.copyhdf5Dataset(cls.dataset_file, cls.dataset_file, "test", "test_copy", usedIndices)
        np.testing.assert_array_equal(cls.dataset_file.get("test").value[usedIndices, :], cls.dataset_file.get("test_copy").value)
        cls.assertEqual("Am I copied", cls.dataset_file.get("test_copy").attrs["test_arg"])

    @classmethod
    def tearDownClass(cls):
        os.remove("Code/Tests/temp_tests/test_copy.hdf5")
        # for dir in os.listdir("Code/Tests/temp_tests"):print(dir)
        # import pdb;pdb.set_trace()
        os.rmdir("Code/Tests/temp_tests")



class Test_filterViews(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.views = ["test_view_1", "test_view_2"]
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        cls.dataset_file = h5py.File("Code/Tests/temp_tests/test_copy.hdf5", "w")
        cls.metadata_group = cls.dataset_file.create_group("Metadata")
        cls.metadata_group.attrs["nbView"] = 4

        for i in range(4):
            cls.dataset = cls.dataset_file.create_dataset("View"+str(i),
                                                          data=cls.random_state.randint(0, 100, (10, 20)))
            cls.dataset.attrs["name"] = "test_view_"+str(i)

    def test_simple_filter(cls):
        cls.temp_dataset_file = h5py.File("Code/Tests/temp_tests/test_copy_temp.hdf5", "w")
        cls.dataset_file.copy("Metadata", cls.temp_dataset_file)
        GetMultiviewDb.filterViews(cls.dataset_file, cls.temp_dataset_file, cls.views, np.arange(10))
        cls.assertEqual(cls.dataset_file.get("View1").attrs["name"],
                        cls.temp_dataset_file.get("View0").attrs["name"])
        np.testing.assert_array_equal(cls.dataset_file.get("View2").value, cls.temp_dataset_file.get("View1").value)
        cls.assertEqual(cls.temp_dataset_file.get("Metadata").attrs["nbView"], 2)

    def test_filter_view_and_examples(cls):
        cls.temp_dataset_file = h5py.File("Code/Tests/temp_tests/test_copy_temp.hdf5", "w")
        cls.dataset_file.copy("Metadata", cls.temp_dataset_file)
        usedIndices = cls.random_state.choice(10, 6, replace=False)
        GetMultiviewDb.filterViews(cls.dataset_file, cls.temp_dataset_file, cls.views, usedIndices)
        np.testing.assert_array_equal(cls.dataset_file.get("View1").value[usedIndices, :],
                                      cls.temp_dataset_file.get("View0").value)

    @classmethod
    def tearDownClass(cls):
        os.remove("Code/Tests/temp_tests/test_copy.hdf5")
        os.remove("Code/Tests/temp_tests/test_copy_temp.hdf5")
        os.rmdir("Code/Tests/temp_tests")

#
class Test_filterLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.labelsSet = set(range(4))
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        cls.fullLabels = cls.random_state.randint(0,4,10)
        cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]

    def test_simple(cls):
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.filterLabels(cls.labelsSet,
                                                  cls.askedLabelsNamesSet,
                                                  cls.fullLabels,
                                                  cls.availableLabelsNames,
                                                  cls.askedLabelsNames)
        cls.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
        np.testing.assert_array_equal(newLabels, np.array([1,1,0]))

    def test_biclasse(cls):
        cls.labelsSet = {0,1}
        cls.fullLabels = cls.random_state.randint(0,2,10)
        cls.availableLabelsNames = ["test_label_0", "test_label_1"]
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.filterLabels(cls.labelsSet,
                                                  cls.askedLabelsNamesSet,
                                                  cls.fullLabels,
                                                  cls.availableLabelsNames,
                                                  cls.askedLabelsNames)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)

    def test_asked_too_many_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3", "chicken_is_heaven"}
        with cls.assertRaises(GetMultiviewDb.DatasetError) as catcher:
            GetMultiviewDb.filterLabels(cls.labelsSet,
                                        cls.askedLabelsNamesSet,
                                        cls.fullLabels,
                                        cls.availableLabelsNames,
                                        cls.askedLabelsNames)
        exception = catcher.exception
        # cls.assertTrue("Asked more labels than available in the dataset. Available labels are : test_label_0, test_label_1, test_label_2, test_label_3" in exception)
    #
    def test_asked_all_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3"}
        cls.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.filterLabels(cls.labelsSet,
                                                  cls.askedLabelsNamesSet,
                                                  cls.fullLabels,
                                                  cls.availableLabelsNames,
                                                  cls.askedLabelsNames)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)


class Test_selectAskedLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        cls.fullLabels = cls.random_state.randint(0, 4, 10)
        cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]

    def test_simple(cls):
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.selectAskedLabels(cls.askedLabelsNamesSet,
                                                       cls.availableLabelsNames,
                                                       cls.askedLabelsNames,
                                                       cls.fullLabels)
        cls.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
        np.testing.assert_array_equal(newLabels, np.array([1, 1, 0]))

    def test_asked_all_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3"}
        cls.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.selectAskedLabels(cls.askedLabelsNamesSet,
                                                       cls.availableLabelsNames,
                                                       cls.askedLabelsNames,
                                                       cls.fullLabels)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)

    def test_asked_unavailable_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3", "chicken_is_heaven"}
        with cls.assertRaises(GetMultiviewDb.DatasetError) as catcher:
            GetMultiviewDb.selectAskedLabels(cls.askedLabelsNamesSet,
                                             cls.availableLabelsNames,
                                             cls.askedLabelsNames,
                                             cls.fullLabels)
        exception = catcher.exception
        # cls.assertTrue("Asked labels are not all available in the dataset" in exception)


class Test_getAllLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.fullLabels = cls.random_state.randint(0, 4, 10)
        cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]

    def test_simple(cls):
        newLabels, newLabelsNames, usedIndices = GetMultiviewDb.getAllLabels(cls.fullLabels, cls.availableLabelsNames)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)


class Test_fillLabelNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.NB_CLASS = 2
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]
        cls.randomState = np.random.RandomState(42)
        cls.availableLabelsNames = ["test_label_"+str(_) for _ in range(40)]

    def test_simple(cls):
        askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(cls.NB_CLASS,
                                                                              cls.askedLabelsNames,
                                                                              cls.randomState,
                                                                              cls.availableLabelsNames)
        cls.assertEqual(askedLabelsNames, cls.askedLabelsNames)
        cls.assertEqual(askedLabelsNamesSet, set(cls.askedLabelsNames))

    def test_missing_labels_names(cls):
        cls.NB_CLASS = 39
        askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(cls.NB_CLASS,
                                                                              cls.askedLabelsNames,
                                                                              cls.randomState,
                                                                              cls.availableLabelsNames)

        cls.assertEqual(askedLabelsNames, ['test_label_1', 'test_label_3', 'test_label_35', 'test_label_38', 'test_label_6', 'test_label_15', 'test_label_32', 'test_label_28', 'test_label_8', 'test_label_29', 'test_label_26', 'test_label_17', 'test_label_19', 'test_label_10', 'test_label_18', 'test_label_14', 'test_label_21', 'test_label_11', 'test_label_34', 'test_label_0', 'test_label_27', 'test_label_7', 'test_label_13', 'test_label_2', 'test_label_39', 'test_label_23', 'test_label_4', 'test_label_31', 'test_label_37', 'test_label_5', 'test_label_36', 'test_label_25', 'test_label_33', 'test_label_12', 'test_label_24', 'test_label_20', 'test_label_22', 'test_label_9', 'test_label_16'])
        cls.assertEqual(askedLabelsNamesSet, set(["test_label_"+str(_) for _ in range(30)]+["test_label_"+str(31+_) for _ in range(9)]))

    def test_too_many_label_names(cls):
        cls.NB_CLASS = 2
        cls.askedLabelsNames = ["test_label_1", "test_label_3", "test_label_4", "test_label_6"]
        askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(cls.NB_CLASS,
                                                                              cls.askedLabelsNames,
                                                                              cls.randomState,
                                                                              cls.availableLabelsNames)
        cls.assertEqual(askedLabelsNames, ["test_label_3", "test_label_6"])
        cls.assertEqual(askedLabelsNamesSet, {"test_label_3", "test_label_6"})


class Test_allAskedLabelsAreAvailable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        cls.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]

    def test_asked_available_labels(cls):
        cls.assertTrue(GetMultiviewDb.allAskedLabelsAreAvailable(cls.askedLabelsNamesSet,cls.availableLabelsNames))

    def test_asked_unavailable_label(cls):
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3", "chicken_is_heaven"}
        cls.assertFalse(GetMultiviewDb.allAskedLabelsAreAvailable(cls.askedLabelsNamesSet,cls.availableLabelsNames))


class Test_getClasses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)

    def test_multiclass(cls):
        labelsSet = GetMultiviewDb.getClasses(cls.random_state.randint(0,5,30))
        cls.assertEqual(labelsSet, {0,1,2,3,4})

    def test_biclass(cls):
        labelsSet = GetMultiviewDb.getClasses(cls.random_state.randint(0,2,30))
        cls.assertEqual(labelsSet, {0,1})

    def test_one_class(cls):
        with cls.assertRaises(GetMultiviewDb.DatasetError) as catcher:
            GetMultiviewDb.getClasses(np.zeros(30,dtype=int))
        exception = catcher.exception
        # cls.assertTrue("Dataset must have at least two different labels" in exception)


class Test_getClassicDBhdf5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        cls.dataset_file = h5py.File("Code/Tests/temp_tests/test_dataset.hdf5", "w")
        cls.pathF = "Code/Tests/temp_tests/"
        cls.nameDB = "test_dataset"
        cls.NB_CLASS = 2
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]
        cls.random_state = np.random.RandomState(42)
        cls.views = ["test_view_1", "test_view_2"]
        cls.metadata_group = cls.dataset_file.create_group("Metadata")
        cls.metadata_group.attrs["nbView"] = 4
        cls.labels_dataset = cls.dataset_file.create_dataset("Labels", data=cls.random_state.randint(0,4,10))
        cls.labels_dataset.attrs["names"] = ["test_label_0".encode(), "test_label_1".encode(), "test_label_2".encode(), "test_label_3".encode()]

        for i in range(4):
            cls.dataset = cls.dataset_file.create_dataset("View" + str(i),
                                                          data=cls.random_state.randint(0, 100, (10, 20)))
            cls.dataset.attrs["name"] = "test_view_" + str(i)

    def test_simple(cls):
        dataset_file, labels_dictionnary = GetMultiviewDb.getClassicDBhdf5(cls.views, cls.pathF, cls.nameDB,
                                                                           cls.NB_CLASS, cls.askedLabelsNames,
                                                                           cls.random_state)
    @classmethod
    def tearDownClass(cls):
        os.remove("Code/Tests/temp_tests/test_dataset_temp_view_label_select.hdf5")
        os.remove("Code/Tests/temp_tests/test_dataset.hdf5")
        dirs = os.listdir("Code/Tests/temp_tests")
        for dir in dirs:
            print(dir)
        os.rmdir("Code/Tests/temp_tests")