import unittest
import argparse
import os
import h5py
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from MonoMultiViewClassifiers.utils import execution


class Test_parseTheArgs(unittest.TestCase):

    def setUp(self):
        self.args = []

    def test_empty_args(self):
        args = execution.parseTheArgs([])
        # print args


class Test_initRandomState(unittest.TestCase):

    def test_random_state_42(self):
        randomState_42 = np.random.RandomState(42)
        randomState = execution.initRandomState("42", "Tests/temp_tests/")
        os.remove("Tests/temp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState.beta(1,100,100),
                                      randomState_42.beta(1,100,100))

    def test_random_state_pickle(self):
        randomState_to_pickle = execution.initRandomState(None, "Tests/temp_tests/")
        pickled_randomState = execution.initRandomState("Tests/temp_tests/randomState.pickle",
                                                        "Tests/temp_tests/")
        os.remove("Tests/temp_tests/randomState.pickle")
        np.testing.assert_array_equal(randomState_to_pickle.beta(1,100,100),
                                      pickled_randomState.beta(1,100,100))


class Test_initLogFile(unittest.TestCase):

    def test_initLogFile(self):
        pass


class Test_genSplits(unittest.TestCase):

    def test_genSplits_no_iter_ratio(self):
        X_indices = np.random.randint(0,500,50)
        labels = np.zeros(500)
        labels[X_indices[:10]] = 1
        foldsObj = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)
        folds = foldsObj.split(X_indices, labels[X_indices])
        for fold in folds:
            train_fold, test_fold = fold
        train_indices = X_indices[train_fold]
        test_indices = X_indices[test_fold]
        self.assertEqual(len(train_indices), 0.8*50)
        self.assertEqual(len(test_indices), 0.2*50)
        for index in test_indices:
            self.assertIn(index, X_indices)
        for index in train_indices:
            self.assertIn(index, X_indices)
        self.assertGreater(len(np.where(labels[train_indices]==0)[0]), 0)
        self.assertGreater(len(np.where(labels[test_indices]==0)[0]), 0)
        self.assertGreater(len(np.where(labels[train_indices]==1)[0]), 0)
        self.assertGreater(len(np.where(labels[test_indices]==1)[0]), 0)
