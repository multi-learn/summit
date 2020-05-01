import unittest
import summit.multiview_platform.metrics as metrics
import pkgutil
import os
from sklearn.metrics._scorer import _BaseScorer

# Tester que chaque metrique a bien les bonnes fonctions qui renvoient bien les bons types d'outputs avec les bons types d'inputs
# Faire de meme pour les differents classifeurs monovues et les differents
# classifeurs multivues


class Test_metric(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test = "a"

    def test_simple(self):
        pkgpath = os.path.dirname(metrics.__file__)
        for _, metric, _ in pkgutil.iter_modules([pkgpath]):
            module = getattr(metrics, metric)
            self.assertTrue(hasattr(module, "score"))
            self.assertTrue(isinstance(module.score([1, 0], [1, 0]), float))
            self.assertTrue(hasattr(module, "get_scorer"))
            self.assertTrue(isinstance(module.get_scorer(), _BaseScorer))
            self.assertTrue(hasattr(module, "get_config"))
            self.assertTrue(isinstance(module.get_config(), str))
