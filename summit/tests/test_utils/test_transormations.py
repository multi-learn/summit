import unittest
import numpy as np


from summit.multiview_platform.utils import transformations


class TestFunctions(unittest.TestCase):

    def test_simple_sign(self):
        trans = transformations.sign_labels(np.zeros(10))
        np.testing.assert_array_equal(np.ones(10) * -1, trans)
        trans = transformations.sign_labels(np.ones(10))
        np.testing.assert_array_equal(np.ones(10), trans)

    def test_simple_unsign(self):
        trans = transformations.unsign_labels(np.ones(10) * -1)
        np.testing.assert_array_equal(np.zeros(10), trans)
        trans = transformations.unsign_labels(np.ones(10).reshape((10, 1)))
        np.testing.assert_array_equal(np.ones(10), trans)
