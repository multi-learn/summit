import unittest
import numpy as np

from ....mono_multi_view_classifiers.multiview_classifiers.additions import diversity_utils


class FakeDataset():

    def __init__(self, views, labels):
        self.nb_views = views.shape[0]
        self.dataset_length = views.shape[2]
        self.views = views
        self.labels = labels

    def get_v(self, view_index, example_indices):
        return self.views[view_index, example_indices]

    def get_nb_class(self, example_indices):
        return np.unique(self.labels[example_indices])

class FakeDivCoupleClf(diversity_utils.CoupleDiversityFusionClassifier):

    def __init__(self, rs, classifier_names=None,
                 classifiers_config=None, monoview_estimators=None):
        super(FakeDivCoupleClf, self).__init__(random_state=rs,
                                               classifier_names=classifier_names,
                                               classifiers_configs=classifiers_config,
                                               monoview_estimators=monoview_estimators)
        self.rs = rs

    def diversity_score(self, a, b, c):
        return self.rs.randint(0,100)


class FakeDivGlobalClf(diversity_utils.GlobalDiversityFusionClassifier):

    def __init__(self, rs, classifier_names=None,
                 classifiers_config=None, monoview_estimators=None):
        super(FakeDivGlobalClf, self).__init__(random_state=rs,
                                               classifier_names=classifier_names,
                                               classifiers_configs=classifiers_config,
                                               monoview_estimators=monoview_estimators)
        self.rs = rs

    def diversity_score(self, a, b, c):
        return self.rs.randint(0,100)

class Test_DiversityFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier_names = ["cb_boost", "decision_tree"]
        cls.classifiers_config = {"cb_boost":{"n_stumps":1, "n_iterations":5}}
        cls.random_state = np.random.RandomState(42)
        cls.y = cls.random_state.randint(0,2,6)
        cls.X = FakeDataset(cls.random_state.randint(0,100,(2,5,6)), cls.y)
        cls.train_indices = [0,1,2,4]
        cls.views_indices = [0,1]

    def test_simple_couple(self):
        clf = FakeDivCoupleClf(self.random_state, classifier_names=self.classifier_names,
                                              classifiers_config=self.classifiers_config)
        clf.fit(self.X, self.y, self.train_indices, self.views_indices)

    def test_simple_global(self):
        clf = FakeDivGlobalClf(self.random_state,
                               classifier_names=self.classifier_names,
                               classifiers_config=self.classifiers_config)
        clf.fit(self.X, self.y, self.train_indices, self.views_indices)