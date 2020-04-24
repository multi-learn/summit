import os
import unittest

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold

from multiview_platform.tests.utils import rm_tmp, tmp_path, test_dataset

from multiview_platform.mono_multi_view_classifiers.multiview import multiview_utils


class FakeMVClassif(multiview_utils.BaseMultiviewClassifier):

    def __init__(self, mc=True):
        self.mc=mc
        pass

    def fit(self, X, y):
        if not self.mc:
            raise ValueError
        else:
            pass



class TestBaseMultiviewClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.mkdir(tmp_path)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_accepts_multiclass(self):
        rs = np.random.RandomState(42)
        accepts = FakeMVClassif().accepts_multi_class(rs)
        self.assertEqual(accepts, True)
        accepts = FakeMVClassif(mc=False).accepts_multi_class(rs)
        self.assertEqual(accepts, False)
