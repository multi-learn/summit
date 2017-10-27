import unittest
import h5py
import numpy as np
import os

from ...MonoMultiViewClassifiers.utils import GetMultiviewDb


class Test_copyhdf5Dataset(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        self.dataset_file = h5py.File("Code/Tests/temp_tests/test_copy.hdf5", "w")
        self.dataset = self.dataset_file.create_dataset("test", data=self.random_state.randint(0,100,(10,20)))
        self.dataset.attrs["test_arg"] = "Am I copied"

    def test_simple_copy(self):
        GetMultiviewDb.copyhdf5Dataset(self.dataset_file, self.dataset_file, "test", "test_copy", np.arange(10))
        np.testing.assert_array_equal(self.dataset_file.get("test").value, self.dataset_file.get("test_copy").value)
        self.assertEqual("Am I copied", self.dataset_file.get("test_copy").attrs["test_arg"])

    def test_copy_only_some_indices(self):
        usedIndices = self.random_state.choice(10,6, replace=False)
        GetMultiviewDb.copyhdf5Dataset(self.dataset_file, self.dataset_file, "test", "test_copy", usedIndices)
        np.testing.assert_array_equal(self.dataset_file.get("test").value[usedIndices, :], self.dataset_file.get("test_copy").value)
        self.assertEqual("Am I copied", self.dataset_file.get("test_copy").attrs["test_arg"])

    def tearDown(self):
        os.remove("Code/Tests/temp_tests/test_copy.hdf5")
        os.rmdir("Code/Tests/temp_tests")



class Test_filterViews(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.views = ["test_view_1", "test_view_2"]
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        self.dataset_file = h5py.File("Code/Tests/temp_tests/test_copy.hdf5", "w")
        self.metadata_group = self.dataset_file.create_group("Metadata")
        self.metadata_group.attrs["nbView"] = 4

        for i in range(4):
            self.dataset = self.dataset_file.create_dataset("View"+str(i),
                                                            data=self.random_state.randint(0, 100, (10, 20)))
            self.dataset.attrs["name"] = "test_view_"+str(i)
        self.temp_dataset_file = h5py.File("Code/Tests/temp_tests/test_copy_temp.hdf5", "w")
        self.dataset_file.copy("Metadata", self.temp_dataset_file)

    def test_simple_filter(self):
        GetMultiviewDb.filterViews(self.dataset_file, self.temp_dataset_file, self.views, np.arange(10))
        self.assertEqual(self.dataset_file.get("View1").attrs["name"],
                         self.temp_dataset_file.get("View0").attrs["name"])
        np.testing.assert_array_equal(self.dataset_file.get("View2").value, self.temp_dataset_file.get("View1").value)
        self.assertEqual(self.temp_dataset_file.get("Metadata").attrs["nbView"], 2)

    def test_filter_view_and_examples(self):
        usedIndices = self.random_state.choice(10, 6, replace=False)
        GetMultiviewDb.filterViews(self.dataset_file, self.temp_dataset_file, self.views, usedIndices)
        np.testing.assert_array_equal(self.dataset_file.get("View1").value[usedIndices, :],
                                      self.temp_dataset_file.get("View0").value)
    def tearDown(self):
        os.remove("Code/Tests/temp_tests/test_copy.hdf5")
        os.remove("Code/Tests/temp_tests/test_copy_temp.hdf5")
        os.rmdir("Code/Tests/temp_tests")


class Test_filterLabels(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.labelsSet = set(range(4))
        self.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        self.fullLabels = self.random_state.randint(0,4,10)
        self.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        self.askedLabelsNames = ["test_label_1", "test_label_3"]

    def test_simple(self):
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.filterLabels(self.labelsSet,
                                                  self.askedLabelsNamesSet,
                                                  self.fullLabels,
                                                  self.availableLabelsNames,
                                                  self.askedLabelsNames)
        self.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
        np.testing.assert_array_equal(newLabels, np.array([1,1,0]))

    def test_biclasse(self):
        self.labelsSet = {0,1}
        self.fullLabels = self.random_state.randint(0,2,10)
        self.availableLabelsNames = ["test_label_0", "test_label_1"]
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.filterLabels(self.labelsSet,
                                                  self.askedLabelsNamesSet,
                                                  self.fullLabels,
                                                  self.availableLabelsNames,
                                                  self.askedLabelsNames)
        self.assertEqual(self.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, self.fullLabels)

    def test_asked_too_many_labels(self):
        self.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3", "chicken_is_heaven"}
        with self.assertRaises(GetMultiviewDb.DatasetError) as catcher:
            GetMultiviewDb.filterLabels(self.labelsSet,
                                        self.askedLabelsNamesSet,
                                        self.fullLabels,
                                        self.availableLabelsNames,
                                        self.askedLabelsNames)
        exception = catcher.exception
        self.assertTrue("Asked more labels than available in the dataset. Available labels are : test_label_0, test_label_1, test_label_2, test_label_3" in exception)

    def test_asked_all_labels(self):
        self.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3"}
        self.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.filterLabels(self.labelsSet,
                                                  self.askedLabelsNamesSet,
                                                  self.fullLabels,
                                                  self.availableLabelsNames,
                                                  self.askedLabelsNames)
        self.assertEqual(self.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, self.fullLabels)


class Test_selectAskedLabels(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        self.fullLabels = self.random_state.randint(0, 4, 10)
        self.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        self.askedLabelsNames = ["test_label_1", "test_label_3"]

    def test_simple(self):
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.selectAskedLabels(self.askedLabelsNamesSet,
                                                       self.availableLabelsNames,
                                                       self.askedLabelsNames,
                                                       self.fullLabels)
        self.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
        np.testing.assert_array_equal(newLabels, np.array([1, 1, 0]))

    def test_asked_all_labels(self):
        self.askedLabelsNamesSet = {"test_label_0", "test_label_1", "test_label_2", "test_label_3"}
        self.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]
        newLabels, \
        newLabelsNames, \
        usedIndices = GetMultiviewDb.selectAskedLabels(self.askedLabelsNamesSet,
                                                       self.availableLabelsNames,
                                                       self.askedLabelsNames,
                                                       self.fullLabels)
        self.assertEqual(self.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, self.fullLabels)

    def test_asked_unavailable_labels(self):
        self.askedLabelsNamesSet = {"test_label_1", "test_label_3", "chicken_is_heaven"}
        with self.assertRaises(GetMultiviewDb.DatasetError) as catcher:
            GetMultiviewDb.selectAskedLabels(self.askedLabelsNamesSet,
                                             self.availableLabelsNames,
                                             self.askedLabelsNames,
                                             self.fullLabels)
        exception = catcher.exception
        self.assertTrue("Asked labels are not all available in the dataset" in exception)


class Test_getAllLabels(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.fullLabels = self.random_state.randint(0, 4, 10)
        self.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]

    def test_simple(self):
        newLabels, newLabelsNames, usedIndices = GetMultiviewDb.getAllLabels(self.fullLabels, self.availableLabelsNames)
        self.assertEqual(self.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, self.fullLabels)


class Test_fillLabelNames(unittest.TestCase):

    def setUp(self):
        self.NB_CLASS = 2
        self.askedLabelsNames = ["test_label_1", "test_label_3"]
        self.randomState = np.random.RandomState(42)
        self.availableLabelsNames = ["test_label_"+str(_) for _ in range(40)]

    def test_simple(self):
        askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(self.NB_CLASS,
                                                                              self.askedLabelsNames,
                                                                              self.randomState,
                                                                              self.availableLabelsNames)
        self.assertEqual(askedLabelsNames, self.askedLabelsNames)
        self.assertEqual(askedLabelsNamesSet, set(self.askedLabelsNames))

    def test_missing_labels_names(self):
        self.NB_CLASS = 39
        askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(self.NB_CLASS,
                                                                              self.askedLabelsNames,
                                                                              self.randomState,
                                                                              self.availableLabelsNames)

        self.assertEqual(askedLabelsNames, ['test_label_1', 'test_label_3', 'test_label_35', 'test_label_38', 'test_label_6', 'test_label_15', 'test_label_32', 'test_label_28', 'test_label_8', 'test_label_29', 'test_label_26', 'test_label_17', 'test_label_19', 'test_label_10', 'test_label_18', 'test_label_14', 'test_label_21', 'test_label_11', 'test_label_34', 'test_label_0', 'test_label_27', 'test_label_7', 'test_label_13', 'test_label_2', 'test_label_39', 'test_label_23', 'test_label_4', 'test_label_31', 'test_label_37', 'test_label_5', 'test_label_36', 'test_label_25', 'test_label_33', 'test_label_12', 'test_label_24', 'test_label_20', 'test_label_22', 'test_label_9', 'test_label_16'])
        self.assertEqual(askedLabelsNamesSet, set(["test_label_"+str(_) for _ in range(30)]+["test_label_"+str(31+_) for _ in range(9)]))

    def test_too_many_label_names(self):
        self.NB_CLASS = 2
        self.askedLabelsNames = ["test_label_1", "test_label_3", "test_label_4", "test_label_6"]
        askedLabelsNames, askedLabelsNamesSet = GetMultiviewDb.fillLabelNames(self.NB_CLASS,
                                                                              self.askedLabelsNames,
                                                                              self.randomState,
                                                                              self.availableLabelsNames)
        self.assertEqual(askedLabelsNames, ["test_label_3", "test_label_6"])
        self.assertEqual(askedLabelsNamesSet, set(["test_label_3", "test_label_6"]))


class Test_allAskedLabelsAreAvailable(unittest.TestCase):

    def setUp(self):
        self.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        self.availableLabelsNames = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]

    def test_asked_available_labels(self):
        self.assertTrue(GetMultiviewDb.allAskedLabelsAreAvailable(self.askedLabelsNamesSet,self.availableLabelsNames))

    def test_asked_unavailable_label(self):
        self.askedLabelsNamesSet = {"test_label_1", "test_label_3", "chicken_is_heaven"}
        self.assertFalse(GetMultiviewDb.allAskedLabelsAreAvailable(self.askedLabelsNamesSet,self.availableLabelsNames))


class Test_getClasses(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)

    def test_multiclass(self):
        labelsSet = GetMultiviewDb.getClasses(self.random_state.randint(0,5,30))
        self.assertEqual(labelsSet, {0,1,2,3,4})

    def test_biclass(self):
        labelsSet = GetMultiviewDb.getClasses(self.random_state.randint(0,2,30))
        self.assertEqual(labelsSet, {0,1})

    def test_one_class(self):
        with self.assertRaises(GetMultiviewDb.DatasetError) as catcher:
            GetMultiviewDb.getClasses(np.zeros(30,dtype=int))
        exception = catcher.exception
        self.assertTrue("Dataset must have at least two different labels" in exception)


class Test_getClassicDBhdf5(unittest.TestCase):

    def setUp(self):
        if not os.path.exists("Code/Tests/temp_tests"):
            os.mkdir("Code/Tests/temp_tests")
        self.dataset_file = h5py.File("Code/Tests/temp_tests/test_dataset.hdf5", "w")
        self.pathF = "Code/Tests/temp_tests/"
        self.nameDB = "test_dataset"
        self.NB_CLASS = 2
        self.askedLabelsNames = ["test_label_1", "test_label_3"]
        self.random_state = np.random.RandomState(42)
        self.views = ["test_view_1", "test_view_2"]
        self.metadata_group = self.dataset_file.create_group("Metadata")
        self.metadata_group.attrs["nbView"] = 4
        self.labels_dataset = self.dataset_file.create_dataset("Labels", data=self.random_state.randint(0,4,10))
        self.labels_dataset.attrs["names"] = ["test_label_0", "test_label_1", "test_label_2", "test_label_3"]

        for i in range(4):
            self.dataset = self.dataset_file.create_dataset("View" + str(i),
                                                            data=self.random_state.randint(0, 100, (10, 20)))
            self.dataset.attrs["name"] = "test_view_" + str(i)

    def test_simple(self):
        dataset_file, labels_dictionnary = GetMultiviewDb.getClassicDBhdf5(self.views, self.pathF, self.nameDB,
                                                                           self.NB_CLASS, self.askedLabelsNames,
                                                                           self.random_state)

    def tearDown(self):
        os.remove("Code/Tests/temp_tests/test_dataset_temp_view_label_select.hdf5")
        os.remove("Code/Tests/temp_tests/test_dataset.hdf5")
        dirs = os.listdir("Code/Tests/temp_tests")
        for dir in dirs:
            print(dir)
        os.rmdir("Code/Tests/temp_tests")