import os
import unittest

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiview_platform.tests.utils import rm_tmp, tmp_path, test_dataset


from multiview_platform.mono_multi_view_classifiers.utils.dataset import HDF5Dataset
from multiview_platform.mono_multi_view_classifiers.utils import hyper_parameter_search
from multiview_platform.mono_multi_view_classifiers.multiview_classifiers import weighted_linear_early_fusion


class Test_randomized_search(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.random_state = np.random.RandomState(42)
        cls.view_weights = [0.5, 0.5]
        os.mkdir(tmp_path)
        cls.dataset_file = h5py.File(
            tmp_path+"test_file.hdf5", "w")
        cls.labels = cls.dataset_file.create_dataset("Labels",
                                                     data=np.array(
                                                         [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, ]))
        cls.view0_data = cls.random_state.randint(1, 10, size=(10, 4))
        view0 = cls.dataset_file.create_dataset("View0",
                                                data=cls.view0_data)
        view0.attrs["sparse"] = False
        view0.attrs["name"] = "ViewN0"
        cls.view1_data = cls.random_state.randint(1, 10, size=(10, 4))
        view1 = cls.dataset_file.create_dataset("View1",
                                                data=cls.view1_data)
        view1.attrs["sparse"] = False
        view1.attrs["name"] = "ViewN1"
        metaDataGrp = cls.dataset_file.create_group("Metadata")
        metaDataGrp.attrs["nbView"] = 2
        metaDataGrp.attrs["nbClass"] = 2
        metaDataGrp.attrs["datasetLength"] = 10
        cls.monoview_classifier_name = "decision_tree"
        cls.monoview_classifier_config = {"max_depth": 1,
                                          "criterion": "gini",
                                          "splitter": "best"}
        cls.k_folds = StratifiedKFold(n_splits=3, random_state=cls.random_state,
                                      shuffle=True)
        cls.learning_indices = np.array([1,2,3,4, 5,6,7,8,9])
        cls.dataset = HDF5Dataset(hdf5_file=cls.dataset_file)

    @classmethod
    def tearDownClass(cls):
        cls.dataset_file.close()
        rm_tmp()


    def test_simple(self):
        best_params, _, params, scores = hyper_parameter_search.randomized_search(
            self.dataset, self.labels[()], "multiview", self.random_state, tmp_path,
            weighted_linear_early_fusion, "WeightedLinearEarlyFusion", self.k_folds,
        1, ["accuracy_score", None], 2, {}, learning_indices=self.learning_indices)
        self.assertIsInstance(best_params, dict)

from sklearn.base import BaseEstimator

class FakeEstim(BaseEstimator):
    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y,):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

class FakeEstimMV(BaseEstimator):
    def __init__(self, param1=None, param2=None):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y,train_indices=None, view_indices=None):
        self.y = y
        return self

    def predict(self, X, example_indices=None, view_indices=None):
        if self.param1=="return exact":
            return self.y[example_indices]
        else:
            return np.zeros(example_indices.shape[0])

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold

class Test_MultiviewCompatibleRandomizedSearchCV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n_splits=2
        cls.estimator = FakeEstim()
        cls.param_distributions = {"param1":[10,100], "param2":[11, 101]}
        cls.n_iter = 4
        cls.refit = True
        cls.n_jobs = 1
        cls.scoring = make_scorer(accuracy_score, )
        cls.cv = StratifiedKFold(n_splits=n_splits, )
        cls.random_state = np.random.RandomState(42)
        cls.learning_indices = np.array([0,1,2, 3, 4,])
        cls.view_indices = None
        cls.framework = "monoview"
        cls.equivalent_draws = False
        cls.X = cls.random_state.randint(0,100, (10,11))
        cls.y = cls.random_state.randint(0,2, 10)

    def test_simple(self):
        hyper_parameter_search.MultiviewCompatibleRandomizedSearchCV(
            self.estimator, self.param_distributions, n_iter=self.n_iter,
            refit=self.refit, n_jobs=self.n_jobs, scoring=self.scoring, cv=self.cv,
            random_state=self.random_state,
            learning_indices=self.learning_indices, view_indices=self.view_indices,
            framework=self.framework,
            equivalent_draws=self.equivalent_draws
        )

    def test_fit(self):
        RSCV = hyper_parameter_search.MultiviewCompatibleRandomizedSearchCV(
            self.estimator, self.param_distributions, n_iter=self.n_iter,
            refit=self.refit, n_jobs=self.n_jobs, scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            learning_indices=self.learning_indices,
            view_indices=self.view_indices,
            framework=self.framework,
            equivalent_draws=self.equivalent_draws
        )
        RSCV.fit(self.X, self.y, )
        tested_param1 = np.ma.masked_array(data=[10,10,100,100],
                     mask=[False, False, False, False])
        np.testing.assert_array_equal(RSCV.cv_results_['param_param1'],
                                      tested_param1)

    def test_fit_multiview(self):
        RSCV = hyper_parameter_search.MultiviewCompatibleRandomizedSearchCV(
            FakeEstimMV(), self.param_distributions, n_iter=self.n_iter,
            refit=self.refit, n_jobs=self.n_jobs, scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            learning_indices=self.learning_indices,
            view_indices=self.view_indices,
            framework="multiview",
            equivalent_draws=self.equivalent_draws
        )
        RSCV.fit(test_dataset, self.y, )
        self.assertEqual(RSCV.n_iter, self.n_iter)

    def test_fit_multiview_equiv(self):
        self.n_iter=1
        RSCV = hyper_parameter_search.MultiviewCompatibleRandomizedSearchCV(
            FakeEstimMV(), self.param_distributions, n_iter=self.n_iter,
            refit=self.refit, n_jobs=self.n_jobs, scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            learning_indices=self.learning_indices,
            view_indices=self.view_indices,
            framework="multiview",
            equivalent_draws=True
        )
        RSCV.fit(test_dataset, self.y, )
        self.assertEqual(RSCV.n_iter, self.n_iter*test_dataset.nb_view)

    def test_gets_good_params(self):
        self.param_distributions["param1"].append('return exact')
        self.n_iter=6
        RSCV = hyper_parameter_search.MultiviewCompatibleRandomizedSearchCV(
            FakeEstimMV(), self.param_distributions, n_iter=self.n_iter,
            refit=self.refit, n_jobs=self.n_jobs, scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            learning_indices=self.learning_indices,
            view_indices=self.view_indices,
            framework="multiview",
            equivalent_draws=False
        )
        RSCV.fit(test_dataset, self.y, )
        self.assertEqual(RSCV.best_params_["param1"], "return exact")


# if __name__ == '__main__':
#     # unittest.main()
#     suite = unittest.TestLoader().loadTestsFromTestCase(Test_randomized_search)
#     unittest.TextTestRunner(verbosity=2).run(suite)