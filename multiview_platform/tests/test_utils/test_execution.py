import os
import unittest

import numpy as np

from ..utils import rm_tmp

from ...mono_multi_view_classifiers.utils import execution


class Test_parseTheArgs(unittest.TestCase):

    def setUp(self):
        self.args = []

    def test_empty_args(self):
        args = execution.parse_the_args([])
        # print args


class Test_initStatsIterRandomStates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.statsIter = 1

    def test_one_statiter(cls):
<<<<<<< HEAD
        cls.state = cls.random_state.get_state()[1]
        statsIterRandomStates = execution.initStatsIterRandomStates(
            cls.statsIter, cls.random_state)
=======
        cls.state = cls.randomState.get_state()[1]
        statsIterRandomStates = execution.init_stats_iter_random_states(
            cls.statsIter, cls.randomState)
>>>>>>> 7b3e918b4fb2938657cae3093d95b1bd6fc461d4
        np.testing.assert_array_equal(statsIterRandomStates[0].get_state()[1],
                                      cls.state)

    def test_multiple_iter(cls):
        cls.statsIter = 3
<<<<<<< HEAD
        statsIterRandomStates = execution.initStatsIterRandomStates(
            cls.statsIter, cls.random_state)
=======
        statsIterRandomStates = execution.init_stats_iter_random_states(
            cls.statsIter, cls.randomState)
>>>>>>> 7b3e918b4fb2938657cae3093d95b1bd6fc461d4
        cls.assertAlmostEqual(len(statsIterRandomStates), 3)
        cls.assertNotEqual(statsIterRandomStates[0].randint(5000),
                           statsIterRandomStates[1].randint(5000))
        cls.assertNotEqual(statsIterRandomStates[0].randint(5000),
                           statsIterRandomStates[2].randint(5000))
        cls.assertNotEqual(statsIterRandomStates[2].randint(5000),
                           statsIterRandomStates[1].randint(5000))


class Test_getDatabaseFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.name = "zrtTap"
        cls.type = ".csv"

    def test_simple(cls):
        getDB = execution.get_database_function(cls.name, cls.type)
        from ...mono_multi_view_classifiers.utils.get_multiview_db import \
            get_classic_db_csv
        cls.assertEqual(getDB, get_classic_db_csv)

    def test_hdf5(cls):
        cls.type = ".hdf5"
        getDB = execution.get_database_function(cls.name, cls.type)
        from ...mono_multi_view_classifiers.utils.get_multiview_db import \
            get_classic_db_hdf5
        cls.assertEqual(getDB, get_classic_db_hdf5)

    def test_plausible_hdf5(cls):
        cls.name = "plausible"
        cls.type = ".hdf5"
        getDB = execution.get_database_function(cls.name, cls.type)
        from ...mono_multi_view_classifiers.utils.get_multiview_db import \
            get_plausible_db_hdf5
        cls.assertEqual(getDB, get_plausible_db_hdf5)


class Test_initRandomState(unittest.TestCase):

    def setUp(self):
        rm_tmp()
        os.mkdir("multiview_platform/tests/tmp_tests/")

    def tearDown(self):
        os.rmdir("multiview_platform/tests/tmp_tests/")

    def test_random_state_42(self):
        randomState_42 = np.random.RandomState(42)
<<<<<<< HEAD
        random_state = execution.initRandomState("42",
                                                "multiview_platform/tests/temp_tests/")
        os.remove("multiview_platform/tests/temp_tests/random_state.pickle")
        np.testing.assert_array_equal(random_state.beta(1, 100, 100),
                                      randomState_42.beta(1, 100, 100))

    def test_random_state_pickle(self):
        randomState_to_pickle = execution.initRandomState(None,
                                                          "multiview_platform/tests/temp_tests/")
        pickled_randomState = execution.initRandomState(
            "multiview_platform/tests/temp_tests/random_state.pickle",
            "multiview_platform/tests/temp_tests/")
        os.remove("multiview_platform/tests/temp_tests/random_state.pickle")
=======
        randomState = execution.init_random_state("42",
                                                "multiview_platform/tests/tmp_tests/")
        os.remove("multiview_platform/tests/tmp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState.beta(1, 100, 100),
                                      randomState_42.beta(1, 100, 100))

    def test_random_state_pickle(self):
        randomState_to_pickle = execution.init_random_state(None,
                                                          "multiview_platform/tests/tmp_tests/")
        pickled_randomState = execution.init_random_state(
            "multiview_platform/tests/tmp_tests/randomState.pickle",
            "multiview_platform/tests/tmp_tests/")
        os.remove("multiview_platform/tests/tmp_tests/randomState.pickle")
>>>>>>> 7b3e918b4fb2938657cae3093d95b1bd6fc461d4
        np.testing.assert_array_equal(randomState_to_pickle.beta(1, 100, 100),
                                      pickled_randomState.beta(1, 100, 100))


class FakeArg():

    def __init__(self):
        self.name = "zrtTap"
        self.CL_type = ["fromage", "jambon"]
        self.views = ["view1", "view2"]
        self.log = True


# Impossible to test as the main directory is notthe same for the exec and the test
# class Test_initLogFile(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.fakeArgs = FakeArg()
#         cls.timestr = time.strftime("%Y_%m_%d-%H_%M")
#
#     def test_initLogFile(cls):
#         cls.timestr = time.strftime("%Y_%m_%d-%H_%M")
#         execution.initLogFile(cls.fakeArgs)
#         cls.assertIn("zrtTap", os.listdir("mutliview_platform/results"), "Database directory not created")
#         cls.assertIn("started_"+cls.timestr, os.listdir("mutliview_platform/results/zrtTap"),"experimentation dir not created")
#         cls.assertIn(cls.timestr + "-" + ''.join(cls.fakeArgs.CL_type) + "-" + "_".join(
#         cls.fakeArgs.views) + "-" + cls.fakeArgs.name + "-LOG.log", os.listdir("mutliview_platform/results/zrtTap/"+"started_"+cls.timestr), "logfile was not created")
#
#     @classmethod
#     def tearDownClass(cls):
#         shutil.rmtree("multiview_platform/results/zrtTap")
#         pass


class Test_genSplits(unittest.TestCase):

    def setUp(self):
        self.stastIter = 3
        self.statsIterRandomStates = [np.random.RandomState(42 + i + 1) for i in
                                      range(self.stastIter)]
        self.random_state = np.random.RandomState(42)
        self.X_indices = self.random_state.randint(0, 500, 50)
        self.labels = np.zeros(500)
        self.labels[self.X_indices[:10]] = 1
        self.labels[self.X_indices[11:30]] = 2  # To test multiclass
        self.splitRatio = 0.2

    def test_simple(self):
        splits = execution.gen_splits(self.labels, self.splitRatio,
                                     self.statsIterRandomStates)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[1]), 2)
        self.assertEqual(type(splits[1][0]), np.ndarray)
        self.assertAlmostEqual(len(splits[1][0]), 0.8 * 500)
        self.assertAlmostEqual(len(splits[1][1]), 0.2 * 500)
        self.assertGreater(len(np.where(self.labels[splits[1][0]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][0]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][0]] == 2)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]] == 2)[0]), 0)

    def test_genSplits_no_iter(self):
        splits = execution.gen_splits(self.labels, self.splitRatio,
                                     self.statsIterRandomStates)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(type(splits[0][0]), np.ndarray)
        self.assertAlmostEqual(len(splits[0][0]), 0.8 * 500)
        self.assertAlmostEqual(len(splits[0][1]), 0.2 * 500)
        self.assertGreater(len(np.where(self.labels[splits[0][0]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][0]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][0]] == 2)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]] == 2)[0]), 0)


class Test_genKFolds(unittest.TestCase):

    def setUp(self):
        self.statsIter = 2
        self.nbFolds = 5
        self.statsIterRandomStates = [np.random.RandomState(42),
                                      np.random.RandomState(94)]

    def test_genKFolds_iter(self):
        pass


class Test_genDirecortiesNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = "../chicken_is_heaven/"
        cls.stats_iter = 5

    def test_simple_ovo(cls):
        directories = execution.gen_direcorties_names(cls.directory,
                                                    cls.stats_iter)
        cls.assertEqual(len(directories), 5)
        cls.assertEqual(directories[0], "../chicken_is_heaven/iter_1/")
        cls.assertEqual(directories[-1], "../chicken_is_heaven/iter_5/")

    def test_ovo_no_iter(cls):
        cls.stats_iter = 1
        directories = execution.gen_direcorties_names(cls.directory,
                                                    cls.stats_iter)
        cls.assertEqual(len(directories), 1)
        cls.assertEqual(directories[0], "../chicken_is_heaven/")


class Test_genArgumentDictionaries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.labelsDictionary = {0: "yes", 1: "No", 2: "Maybe"}
        cls.direcories = ["Res/iter_1", "Res/iter_2"]
        cls.multiclassLabels = [np.array([0, 1, -100, 1, 0]),
                                np.array([1, 0, -100, 1, 0]),
                                np.array([0, 1, -100, 0, 1])]
        cls.labelsCombinations = [[0, 1], [0, 2], [1, 2]]
        cls.indicesMulticlass = [[[[], []], [[], []], [[], []]], [[], [], []]]
