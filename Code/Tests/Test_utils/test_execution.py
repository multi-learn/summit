import unittest
import argparse
import os
import h5py
import numpy as np

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

    def test_genSplits_no_iter(self):
        pass