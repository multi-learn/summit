import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from multiview_platform.mono_multi_view_classifiers.monoview import monoview_utils


class Test_genTestFoldsPreds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.X_train = cls.random_state.random_sample((31, 10))
        cls.y_train = np.ones(31, dtype=int)
        cls.KFolds = StratifiedKFold(n_splits=3,)

        cls.estimator = DecisionTreeClassifier(max_depth=1)

        cls.y_train[15:] = -1
        # print(cls.X_train)
        # print(cls.y_train)

    def test_simple(cls):
        testFoldsPreds = monoview_utils.gen_test_folds_preds(cls.X_train,
                                                             cls.y_train,
                                                             cls.KFolds,
                                                             cls.estimator)
        cls.assertEqual(testFoldsPreds.shape, (3, 10))
        np.testing.assert_array_equal(testFoldsPreds[0], np.array(
            [ 1,  1, -1, -1,  1,  1, -1,  1, -1,  1]))

