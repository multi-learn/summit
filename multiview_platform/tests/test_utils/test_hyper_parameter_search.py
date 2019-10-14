import os
import unittest

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiview_platform.tests.utils import rm_tmp, tmp_path


from multiview_platform.mono_multi_view_classifiers.utils.dataset import Dataset
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
        cls.k_folds = StratifiedKFold(n_splits=3, random_state=cls.random_state)
        cls.learning_indices = np.array([1,2,3,4, 5,6,7,8,9])
        cls.dataset = Dataset(hdf5_file=cls.dataset_file)

    @classmethod
    def tearDownClass(cls):
        cls.dataset_file.close()
        rm_tmp()


    def test_simple(self):
        best_params, test_folds_preds = hyper_parameter_search.randomized_search(
            self.dataset, self.labels.value, "multiview", self.random_state, tmp_path,
            weighted_linear_early_fusion, "WeightedLinearEarlyFusion", self.k_folds,
        1, ["accuracy_score", None], 2, {}, learning_indices=self.learning_indices)


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_randomized_search)
    unittest.TextTestRunner(verbosity=2).run(suite)