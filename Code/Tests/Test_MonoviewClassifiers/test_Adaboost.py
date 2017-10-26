import unittest
import numpy as np

from ...MonoMultiViewClassifiers.MonoviewClassifiers import Adaboost


class Test_fit(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.dataset = self.random_state.randint(0, 100, (10, 5))
        self.labels = self.random_state.randint(0, 2, 10)
        self.kwargs = {"0": 5}
        self.classifier = Adaboost.fit(self.dataset, self.labels, 42, NB_CORES=1, **self.kwargs)

    def test_fit_kwargs_string(self):
        self.kwargs = {"0": "5"}
        classifier = Adaboost.fit(self.dataset, self.labels, 42, NB_CORES=1, **self.kwargs)
        self.assertEqual(classifier.n_estimators, 5)

    def test_fit_kwargs_int(self):
        self.kwargs = {"0": 5}
        classifier = Adaboost.fit(self.dataset, self.labels, 42, NB_CORES=1, **self.kwargs)
        self.assertEqual(classifier.n_estimators, 5)

    def test_fit_labels(self):
        predicted_labels = self.classifier.predict(self.dataset)
        np.testing.assert_array_equal(predicted_labels, self.labels)

