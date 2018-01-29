import unittest
import argparse
import os
import h5py
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from ...MonoMultiViewClassifiers.utils import execution


class Test_parseTheArgs(unittest.TestCase):

    def setUp(self):
        self.args = []

    def test_empty_args(self):
        args = execution.parseTheArgs([])
        # print args


class Test_initRandomState(unittest.TestCase):

    def setUp(self):
        os.mkdir("multiview_platform/Tests/temp_tests/")

    def tearDown(self):
        os.rmdir("multiview_platform/Tests/temp_tests/")

    def test_random_state_42(self):
        randomState_42 = np.random.RandomState(42)
        randomState = execution.initRandomState("42", "multiview_platform/Tests/temp_tests/")
        os.remove("multiview_platform/Tests/temp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState.beta(1,100,100),
                                      randomState_42.beta(1,100,100))

    def test_random_state_pickle(self):
        randomState_to_pickle = execution.initRandomState(None, "multiview_platform/Tests/temp_tests/")
        pickled_randomState = execution.initRandomState("multiview_platform/Tests/temp_tests/randomState.pickle",
                                                        "multiview_platform/Tests/temp_tests/")
        os.remove("multiview_platform/Tests/temp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState_to_pickle.beta(1,100,100),
                                      pickled_randomState.beta(1,100,100))


class Test_initLogFile(unittest.TestCase):

    def test_initLogFile(self):
        pass


class Test_genSplits(unittest.TestCase):

    def setUp(self):
        self.stastIter = 3
        self.statsIterRandomStates = [np.random.RandomState(42+i+1) for i in range(self.stastIter)]
        self.random_state = np.random.RandomState(42)
        self.X_indices = self.random_state.randint(0,500,50)
        self.labels = np.zeros(500)
        self.labels[self.X_indices[:10]] = 1
        self.labels[self.X_indices[11:30]] = 2  # To test multiclass
        self.splitRatio = 0.2

    def test_simple(self):
        splits = execution.genSplits(self.labels, self.splitRatio, self.statsIterRandomStates)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[1]), 2)
        self.assertEqual(type(splits[1][0]), np.ndarray)
        self.assertAlmostEqual(len(splits[1][0]), 0.8*500)
        self.assertAlmostEqual(len(splits[1][1]), 0.2*500)
        self.assertGreater(len(np.where(self.labels[splits[1][0]]==0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][0]]==1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][0]]==2)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]]==0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]]==1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]]==2)[0]), 0)

    def test_genSplits_no_iter(self):
        splits = execution.genSplits(self.labels, self.splitRatio, self.statsIterRandomStates)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(type(splits[0][0]), np.ndarray)
        self.assertAlmostEqual(len(splits[0][0]), 0.8*500)
        self.assertAlmostEqual(len(splits[0][1]), 0.2*500)
        self.assertGreater(len(np.where(self.labels[splits[0][0]]==0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][0]]==1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][0]]==2)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]]==0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]]==1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]]==2)[0]), 0)


class Test_genKFolds(unittest.TestCase):

    def setUp(self):
        self.statsIter = 2
        self.nbFolds = 5
        self.statsIterRandomStates = [np.random.RandomState(42), np.random.RandomState(94)]

    def test_genKFolds_iter(self):
        pass


class Test_genDirecortiesNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = "../chicken_is_heaven/"
        cls.stats_iter = 5

    def test_simple_ovo(cls):
        directories = execution.genDirecortiesNames(cls.directory, cls.stats_iter)
        cls.assertEqual(len(directories), 5)
        cls.assertEqual(directories[0], "../chicken_is_heaven/iter_1/")
        cls.assertEqual(directories[-1], "../chicken_is_heaven/iter_5/")

    def test_ovo_no_iter(cls):
        cls.stats_iter = 1
        directories = execution.genDirecortiesNames(cls.directory, cls.stats_iter)
        cls.assertEqual(len(directories), 1)
        cls.assertEqual(directories[0], "../chicken_is_heaven/")

class Test_genArgumentDictionaries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.labelsDictionary = {0:"yes", 1:"No", 2:"Maybe"}
        cls.direcories = ["Res/iter_1", "Res/iter_2"]
        cls.multiclassLabels = [np.array([0, 1, -100, 1, 0]),
                                np.array([1, 0, -100, 1, 0]),
                                np.array([0, 1, -100, 0, 1])]
        cls.labelsCombinations = [[0,1], [0,2], [1,2]]
        cls.indicesMulticlass = [[[[], []], [[], []], [[], []]], [[], [], []]]