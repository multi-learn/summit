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

    def test_random_state_42(self):
        randomState_42 = np.random.RandomState(42)
        randomState = execution.initRandomState("42", "Code/Tests/temp_tests/")
        os.remove("Code/Tests/temp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState.beta(1,100,100),
                                      randomState_42.beta(1,100,100))

    def test_random_state_pickle(self):
        randomState_to_pickle = execution.initRandomState(None, "Code/Tests/temp_tests/")
        pickled_randomState = execution.initRandomState("Code/Tests/temp_tests/randomState.pickle",
                                                        "Code/Tests/temp_tests/")
        os.remove("Code/Tests/temp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState_to_pickle.beta(1,100,100),
                                      pickled_randomState.beta(1,100,100))


class Test_initLogFile(unittest.TestCase):

    def test_initLogFile(self):
        pass


class Test_genSplits(unittest.TestCase):

    def setUp(self):
        self.X_indices = np.random.randint(0,500,50)
        self.labels = np.zeros(500)
        self.labels[self.X_indices[:10]] = 1
        self.foldsObj = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)
        self.folds = self.foldsObj.split(self.X_indices, self.labels[self.X_indices])
        for fold in self.folds:
            self.train_fold, self.test_fold = fold
        self.train_indices = self.X_indices[self.train_fold]
        self.test_indices = self.X_indices[self.test_fold]

    def test_genSplits_no_iter_ratio(self):
        self.assertEqual(len(self.train_indices), 0.8*50)
        self.assertEqual(len(self.test_indices), 0.2*50)

    def test_genSplits_no_iter_presence(self):
        for index in self.test_indices:
            self.assertIn(index, self.X_indices)
        for index in self.train_indices:
            self.assertIn(index, self.X_indices)

    def test_genSplits_no_iter_balance(self):
        self.assertGreater(len(np.where(self.labels[self.train_indices]==0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[self.test_indices]==0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[self.train_indices]==1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[self.test_indices]==1)[0]), 0)

class Test_genKFolds(unittest.TestCase):

    def setUp(self):
        self.statsIter = 2
        self.nbFolds = 5
        self.statsIterRandomStates = [np.random.RandomState(42), np.random.RandomState(94)]

    def test_genKFolds_iter(self):
        pass
