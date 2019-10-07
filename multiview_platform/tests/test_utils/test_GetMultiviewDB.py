import os
import unittest

import h5py
import numpy as np

from ...mono_multi_view_classifiers.utils import get_multiview_db
from ..utils import rm_tmp, tmp_path


class Test_copyhdf5Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.random_state = np.random.RandomState(42)
        if not os.path.exists("multiview_platform/tests/tmp_tests"):
            os.mkdir("multiview_platform/tests/tmp_tests")
        cls.dataset_file = h5py.File(
            tmp_path+"test_copy.hdf5", "w")
        cls.dataset = cls.dataset_file.create_dataset("test",
                                                      data=cls.random_state.randint(
                                                          0, 100, (10, 20)))
        cls.dataset.attrs["test_arg"] = "Am I copied"

    def test_simple_copy(cls):
        get_multiview_db.copyhdf5_dataset(cls.dataset_file, cls.dataset_file,
                                       "test", "test_copy_1", np.arange(10))
        np.testing.assert_array_equal(cls.dataset_file.get("test").value,
                                      cls.dataset_file.get("test_copy_1").value)
        cls.assertEqual("Am I copied",
                        cls.dataset_file.get("test_copy_1").attrs["test_arg"])

    def test_copy_only_some_indices(cls):
        usedIndices = cls.random_state.choice(10, 6, replace=False)
        get_multiview_db.copyhdf5_dataset(cls.dataset_file, cls.dataset_file,
                                       "test", "test_copy", usedIndices)
        np.testing.assert_array_equal(
            cls.dataset_file.get("test").value[usedIndices, :],
            cls.dataset_file.get("test_copy").value)
        cls.assertEqual("Am I copied",
                        cls.dataset_file.get("test_copy").attrs["test_arg"])

    @classmethod
    def tearDownClass(cls):
        os.remove(tmp_path+"test_copy.hdf5")
        os.rmdir("multiview_platform/tests/tmp_tests")


class Test_filterViews(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.random_state = np.random.RandomState(42)
        cls.views = ["test_view_1", "test_view_2"]
        if not os.path.exists("multiview_platform/tests/tmp_tests"):
            os.mkdir("multiview_platform/tests/tmp_tests")
        cls.dataset_file = h5py.File(
            tmp_path+"test_copy.hdf5", "w")
        cls.metadata_group = cls.dataset_file.create_group("Metadata")
        cls.metadata_group.attrs["nbView"] = 4

        for i in range(4):
            cls.dataset = cls.dataset_file.create_dataset("View" + str(i),
                                                          data=cls.random_state.randint(
                                                              0, 100, (10, 20)))
            cls.dataset.attrs["name"] = "test_view_" + str(i)

    def test_simple_filter(cls):
        cls.temp_dataset_file = h5py.File(
            tmp_path+"test_copy_temp.hdf5", "w")
        cls.dataset_file.copy("Metadata", cls.temp_dataset_file)
        get_multiview_db.filter_views(cls.dataset_file, cls.temp_dataset_file,
                                     cls.views, np.arange(10))
        cls.assertEqual(cls.dataset_file.get("View1").attrs["name"],
                        cls.temp_dataset_file.get("View0").attrs["name"])
        np.testing.assert_array_equal(cls.dataset_file.get("View2").value,
                                      cls.temp_dataset_file.get("View1").value)
        cls.assertEqual(cls.temp_dataset_file.get("Metadata").attrs["nbView"],
                        2)

    def test_filter_view_and_examples(cls):
        cls.temp_dataset_file = h5py.File(
            tmp_path+"test_copy_temp.hdf5", "w")
        cls.dataset_file.copy("Metadata", cls.temp_dataset_file)
        usedIndices = cls.random_state.choice(10, 6, replace=False)
        get_multiview_db.filter_views(cls.dataset_file, cls.temp_dataset_file,
                                     cls.views, usedIndices)
        np.testing.assert_array_equal(
            cls.dataset_file.get("View1").value[usedIndices, :],
            cls.temp_dataset_file.get("View0").value)
        cls.temp_dataset_file.close()

    @classmethod
    def tearDownClass(cls):
        os.remove(tmp_path+"test_copy.hdf5")
        os.remove(tmp_path+"test_copy_temp.hdf5")
        os.rmdir("multiview_platform/tests/tmp_tests")


#
class Test_filterLabels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.labelsSet = set(range(4))
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        cls.fullLabels = cls.random_state.randint(0, 4, 10)
        cls.availableLabelsNames = ["test_label_0", "test_label_1",
                                    "test_label_2", "test_label_3"]
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]

    def test_simple(cls):
        newLabels, \
        newLabelsNames, \
        usedIndices = get_multiview_db.filter_labels(cls.labelsSet,
                                                    cls.askedLabelsNamesSet,
                                                    cls.fullLabels,
                                                    cls.availableLabelsNames,
                                                    cls.askedLabelsNames)
        cls.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
        np.testing.assert_array_equal(newLabels, np.array([1, 1, 0]))

    def test_biclasse(cls):
        cls.labelsSet = {0, 1}
        cls.fullLabels = cls.random_state.randint(0, 2, 10)
        cls.availableLabelsNames = ["test_label_0", "test_label_1"]
        newLabels, \
        newLabelsNames, \
        usedIndices = get_multiview_db.filter_labels(cls.labelsSet,
                                                    cls.askedLabelsNamesSet,
                                                    cls.fullLabels,
                                                    cls.availableLabelsNames,
                                                    cls.askedLabelsNames)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)

    def test_asked_too_many_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_0", "test_label_1",
                                   "test_label_2", "test_label_3",
                                   "chicken_is_heaven"}
        with cls.assertRaises(get_multiview_db.DatasetError) as catcher:
            get_multiview_db.filter_labels(cls.labelsSet,
                                          cls.askedLabelsNamesSet,
                                          cls.fullLabels,
                                          cls.availableLabelsNames,
                                          cls.askedLabelsNames)
        exception = catcher.exception

    def test_asked_all_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_0", "test_label_1",
                                   "test_label_2", "test_label_3"}
        cls.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2",
                                "test_label_3"]
        newLabels, \
        newLabelsNames, \
        usedIndices = get_multiview_db.filter_labels(cls.labelsSet,
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
        cls.availableLabelsNames = ["test_label_0", "test_label_1",
                                    "test_label_2", "test_label_3"]
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]

    def test_simple(cls):
        newLabels, \
        newLabelsNames, \
        usedIndices = get_multiview_db.select_asked_labels(cls.askedLabelsNamesSet,
                                                         cls.availableLabelsNames,
                                                         cls.askedLabelsNames,
                                                         cls.fullLabels)
        cls.assertEqual(["test_label_1", "test_label_3"], newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.array([1, 5, 9]))
        np.testing.assert_array_equal(newLabels, np.array([1, 1, 0]))

    def test_asked_all_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_0", "test_label_1",
                                   "test_label_2", "test_label_3"}
        cls.askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2",
                                "test_label_3"]
        newLabels, \
        newLabelsNames, \
        usedIndices = get_multiview_db.select_asked_labels(cls.askedLabelsNamesSet,
                                                         cls.availableLabelsNames,
                                                         cls.askedLabelsNames,
                                                         cls.fullLabels)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)

    def test_asked_unavailable_labels(cls):
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3",
                                   "chicken_is_heaven"}
        with cls.assertRaises(get_multiview_db.DatasetError) as catcher:
            get_multiview_db.select_asked_labels(cls.askedLabelsNamesSet,
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
        cls.availableLabelsNames = ["test_label_0", "test_label_1",
                                    "test_label_2", "test_label_3"]

    def test_simple(cls):
        newLabels, newLabelsNames, usedIndices = get_multiview_db.get_all_labels(
            cls.fullLabels, cls.availableLabelsNames)
        cls.assertEqual(cls.availableLabelsNames, newLabelsNames)
        np.testing.assert_array_equal(usedIndices, np.arange(10))
        np.testing.assert_array_equal(newLabels, cls.fullLabels)


class Test_fillLabelNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.NB_CLASS = 2
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]
        cls.random_state = np.random.RandomState(42)
        cls.availableLabelsNames = ["test_label_" + str(_) for _ in range(40)]

    def test_simple(cls):
        askedLabelsNames, askedLabelsNamesSet = get_multiview_db.fill_label_names(
            cls.NB_CLASS,
            cls.askedLabelsNames,
            cls.random_state,
            cls.availableLabelsNames)
        cls.assertEqual(askedLabelsNames, cls.askedLabelsNames)
        cls.assertEqual(askedLabelsNamesSet, set(cls.askedLabelsNames))

    def test_missing_labels_names(cls):
        cls.NB_CLASS = 39
        askedLabelsNames, askedLabelsNamesSet = get_multiview_db.fill_label_names(
            cls.NB_CLASS,
            cls.askedLabelsNames,
            cls.random_state,
            cls.availableLabelsNames)

        cls.assertEqual(askedLabelsNames,
                        ['test_label_1', 'test_label_3', 'test_label_35',
                         'test_label_38', 'test_label_6', 'test_label_15',
                         'test_label_32', 'test_label_28', 'test_label_8',
                         'test_label_29', 'test_label_26', 'test_label_17',
                         'test_label_19', 'test_label_10', 'test_label_18',
                         'test_label_14', 'test_label_21', 'test_label_11',
                         'test_label_34', 'test_label_0', 'test_label_27',
                         'test_label_7', 'test_label_13', 'test_label_2',
                         'test_label_39', 'test_label_23', 'test_label_4',
                         'test_label_31', 'test_label_37', 'test_label_5',
                         'test_label_36', 'test_label_25', 'test_label_33',
                         'test_label_12', 'test_label_24', 'test_label_20',
                         'test_label_22', 'test_label_9', 'test_label_16'])
        cls.assertEqual(askedLabelsNamesSet, set(
            ["test_label_" + str(_) for _ in range(30)] + [
                "test_label_" + str(31 + _) for _ in range(9)]))

    def test_too_many_label_names(cls):
        cls.NB_CLASS = 2
        cls.askedLabelsNames = ["test_label_1", "test_label_3", "test_label_4",
                                "test_label_6"]
        askedLabelsNames, askedLabelsNamesSet = get_multiview_db.fill_label_names(
            cls.NB_CLASS,
            cls.askedLabelsNames,
            cls.random_state,
            cls.availableLabelsNames)
        cls.assertEqual(askedLabelsNames, ["test_label_3", "test_label_6"])
        cls.assertEqual(askedLabelsNamesSet, {"test_label_3", "test_label_6"})


class Test_allAskedLabelsAreAvailable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3"}
        cls.availableLabelsNames = ["test_label_0", "test_label_1",
                                    "test_label_2", "test_label_3"]

    def test_asked_available_labels(cls):
        cls.assertTrue(
            get_multiview_db.all_asked_labels_are_available(cls.askedLabelsNamesSet,
                                                        cls.availableLabelsNames))

    def test_asked_unavailable_label(cls):
        cls.askedLabelsNamesSet = {"test_label_1", "test_label_3",
                                   "chicken_is_heaven"}
        cls.assertFalse(
            get_multiview_db.all_asked_labels_are_available(cls.askedLabelsNamesSet,
                                                        cls.availableLabelsNames))


class Test_getClasses(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)

    def test_multiclass(cls):
        labelsSet = get_multiview_db.get_classes(
            cls.random_state.randint(0, 5, 30))
        cls.assertEqual(labelsSet, {0, 1, 2, 3, 4})

    def test_biclass(cls):
        labelsSet = get_multiview_db.get_classes(
            cls.random_state.randint(0, 2, 30))
        cls.assertEqual(labelsSet, {0, 1})

    def test_one_class(cls):
        with cls.assertRaises(get_multiview_db.DatasetError) as catcher:
            get_multiview_db.get_classes(np.zeros(30, dtype=int))
        exception = catcher.exception
        # cls.assertTrue("Dataset must have at least two different labels" in exception)


class Test_getClassicDBhdf5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        if not os.path.exists("multiview_platform/tests/tmp_tests"):
            os.mkdir("multiview_platform/tests/tmp_tests")
        cls.dataset_file = h5py.File(
            tmp_path+"test_dataset.hdf5", "w")
        cls.pathF = tmp_path
        cls.nameDB = "test_dataset"
        cls.NB_CLASS = 2
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]
        cls.random_state = np.random.RandomState(42)
        cls.views = ["test_view_1", "test_view_3"]
        cls.metadata_group = cls.dataset_file.create_group("Metadata")
        cls.metadata_group.attrs["nbView"] = 4
        cls.labels_dataset = cls.dataset_file.create_dataset("Labels",
                                                             data=cls.random_state.randint(
                                                                 0, 4, 10))
        cls.labels_dataset.attrs["names"] = ["test_label_0".encode(),
                                             "test_label_1".encode(),
                                             "test_label_2".encode(),
                                             "test_label_3".encode()]

        for i in range(4):
            cls.dataset = cls.dataset_file.create_dataset("View" + str(i),
                                                          data=cls.random_state.randint(
                                                              0, 100, (10, 20)))
            cls.dataset.attrs["name"] = "test_view_" + str(i)

    def test_simple(cls):
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(
            cls.views, cls.pathF, cls.nameDB,
            cls.NB_CLASS, cls.askedLabelsNames,
            cls.random_state)
        cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_1", 1: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)
        np.testing.assert_array_equal(dataset_file.get("View0").value,
                                      cls.dataset_file.get("View1").value[
                                      np.array([1, 5, 9]), :])

    def test_all_labels_asked(cls):
        askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2",
                            "test_label_3"]
        NB_CLASS = 4
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(
            cls.views, cls.pathF, cls.nameDB,
            NB_CLASS, askedLabelsNames,
            cls.random_state)
        cls.assertEqual(dataset_name, 'test_dataset_temp_view_label_select')
        cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_0", 1: "test_label_1",
                         2: "test_label_2", 3: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 10)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 4)
        np.testing.assert_array_equal(dataset_file.get("View0").value,
                                      cls.dataset_file.get("View1").value)

    def test_all_views_asked(cls):
        views = ["test_view_0", "test_view_1", "test_view_2", "test_view_3"]
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(views,
                                                                                          cls.pathF,
                                                                                          cls.nameDB,
                                                                                          cls.NB_CLASS,
                                                                                          cls.askedLabelsNames,
                                                                                          cls.random_state)
        for viewIndex in range(4):
            np.testing.assert_array_equal(
                dataset_file.get("View" + str(viewIndex)).value,
                cls.dataset_file.get("View" + str(viewIndex)).value[
                np.array([1, 5, 9]), :])
            cls.assertEqual(
                dataset_file.get("View" + str(viewIndex)).attrs["name"],
                "test_view_" + str(viewIndex))
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_1", 1: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 4)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)

    def test_asked_the_whole_dataset(cls):
        askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2",
                            "test_label_3"]
        NB_CLASS = 4
        views = ["test_view_0", "test_view_1", "test_view_2", "test_view_3"]
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_hdf5(views,
                                                                                          cls.pathF,
                                                                                          cls.nameDB,
                                                                                          NB_CLASS,
                                                                                          askedLabelsNames,
                                                                                          cls.random_state)
        for viewIndex in range(4):
            np.testing.assert_array_equal(
                dataset_file.get("View" + str(viewIndex)).value,
                cls.dataset_file.get("View" + str(viewIndex)))
            cls.assertEqual(
                dataset_file.get("View" + str(viewIndex)).attrs["name"],
                "test_view_" + str(viewIndex))
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_0", 1: "test_label_1",
                         2: "test_label_2", 3: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 10)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 4)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 4)

    @classmethod
    def tearDownClass(cls):
        os.remove(
            tmp_path+"test_dataset_temp_view_label_select.hdf5")
        os.remove(tmp_path+"test_dataset.hdf5")
        dirs = os.listdir("multiview_platform/tests/tmp_tests")
        for dir in dirs:
            print(dir)
        os.rmdir("multiview_platform/tests/tmp_tests")


class Test_getClassicDBcsv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        if not os.path.exists("multiview_platform/tests/tmp_tests"):
            os.mkdir("multiview_platform/tests/tmp_tests")
        cls.pathF = tmp_path
        cls.NB_CLASS = 2
        cls.nameDB = "test_dataset"
        cls.askedLabelsNames = ["test_label_1", "test_label_3"]
        cls.random_state = np.random.RandomState(42)
        cls.views = ["test_view_1", "test_view_3"]
        np.savetxt(cls.pathF + cls.nameDB + "-labels-names.csv",
                   np.array(["test_label_0", "test_label_1",
                             "test_label_2", "test_label_3"]), fmt="%s",
                   delimiter=",")
        np.savetxt(cls.pathF + cls.nameDB + "-labels.csv",
                   cls.random_state.randint(0, 4, 10), delimiter=",")
        os.mkdir(cls.pathF + "Views")
        cls.datas = []
        for i in range(4):
            data = cls.random_state.randint(0, 100, (10, 20))
            np.savetxt(cls.pathF + "Views/test_view_" + str(i) + ".csv",
                       data, delimiter=",")
            cls.datas.append(data)

    def test_simple(cls):
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_csv(
            cls.views, cls.pathF, cls.nameDB,
            cls.NB_CLASS, cls.askedLabelsNames,
            cls.random_state, delimiter=",")
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
        cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
        cls.assertEqual(dataset_file.get("View0").attrs["name"], "test_view_1")
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_1", 1: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)
        np.testing.assert_array_equal(dataset_file.get("View0").value,
                                      cls.datas[1][np.array([1, 5, 9]), :])

    def test_all_views_asked(cls):
        views = ["test_view_0", "test_view_1", "test_view_2", "test_view_3"]
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_csv(views,
                                                                                         cls.pathF,
                                                                                         cls.nameDB,
                                                                                         cls.NB_CLASS,
                                                                                         cls.askedLabelsNames,
                                                                                         cls.random_state,
                                                                                         delimiter=",")
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_1", 1: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 3)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 4)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 2)
        cls.assertEqual(dataset_name,'test_dataset_temp_view_label_select')
        for viewIndex in range(4):
            np.testing.assert_array_equal(
                dataset_file.get("View" + str(viewIndex)).value,
                cls.datas[viewIndex][np.array([1, 5, 9]), :])
            cls.assertEqual(
                dataset_file.get("View" + str(viewIndex)).attrs["name"],
                "test_view_" + str(viewIndex))

    def test_all_labels_asked(cls):
        askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2",
                            "test_label_3"]
        NB_CLASS = 4
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_csv(
            cls.views, cls.pathF, cls.nameDB,
            NB_CLASS, askedLabelsNames,
            cls.random_state, delimiter=",")
        cls.assertEqual(dataset_file.get("View1").attrs["name"], "test_view_3")
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_0", 1: "test_label_1",
                         2: "test_label_2", 3: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 10)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 2)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 4)
        np.testing.assert_array_equal(dataset_file.get("View0").value,
                                      cls.datas[1])

    def test_asked_the_whole_dataset(cls):
        askedLabelsNames = ["test_label_0", "test_label_1", "test_label_2",
                            "test_label_3"]
        NB_CLASS = 4
        views = ["test_view_0", "test_view_1", "test_view_2", "test_view_3"]
        dataset_file, labels_dictionary, dataset_name = get_multiview_db.get_classic_db_csv(views,
                                                                                         cls.pathF,
                                                                                         cls.nameDB,
                                                                                         NB_CLASS,
                                                                                         askedLabelsNames,
                                                                                         cls.random_state,
                                                                                         delimiter=",")
        for viewIndex in range(4):
            np.testing.assert_array_equal(
                dataset_file.get("View" + str(viewIndex)).value,
                cls.datas[viewIndex])
            cls.assertEqual(
                dataset_file.get("View" + str(viewIndex)).attrs["name"],
                "test_view_" + str(viewIndex))
        cls.assertEqual(labels_dictionary,
                        {0: "test_label_0", 1: "test_label_1",
                         2: "test_label_2", 3: "test_label_3"})
        cls.assertEqual(dataset_file.get("Metadata").attrs["datasetLength"], 10)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbView"], 4)
        cls.assertEqual(dataset_file.get("Metadata").attrs["nbClass"], 4)

    @classmethod
    def tearDownClass(cls):
        for i in range(4):
            os.remove(
                tmp_path+"Views/test_view_" + str(
                    i) + ".csv")
        os.rmdir(tmp_path+"Views")
        os.remove(
            tmp_path+"test_dataset-labels-names.csv")
        os.remove(tmp_path+"test_dataset-labels.csv")
        os.remove(tmp_path+"test_dataset.hdf5")
        os.remove(
            tmp_path+"test_dataset_temp_view_label_select.hdf5")
        for file in os.listdir("multiview_platform/tests/tmp_tests"): print(
            file)
        os.rmdir("multiview_platform/tests/tmp_tests")

class Test_get_plausible_db_hdf5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.path = tmp_path
        cls.nb_class=3
        cls.rs = np.random.RandomState(42)
        cls.nb_view=3
        cls.nb_examples = 5
        cls.nb_features = 4

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        dataset, labels_dict, name = get_multiview_db.get_plausible_db_hdf5(
            "", self.path, "", nb_class=self.nb_class, random_state=self.rs,
            nb_view=3, nb_examples=self.nb_examples,
            nb_features=self.nb_features)
        self.assertEqual(dataset.init_example_indces(), range(5))
        self.assertEqual(dataset.get_nb_class(), self.nb_class)

    def test_two_class(self):
        dataset, labels_dict, name = get_multiview_db.get_plausible_db_hdf5(
            "", self.path, "", nb_class=2, random_state=self.rs,
            nb_view=3, nb_examples=self.nb_examples,
            nb_features=self.nb_features)
        self.assertEqual(dataset.init_example_indces(), range(5))
        self.assertEqual(dataset.get_nb_class(), 2)


