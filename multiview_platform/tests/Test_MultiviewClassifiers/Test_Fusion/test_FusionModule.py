import unittest

from ....mono_multi_view_classifiers.multiview_classifiers.fusion import \
    FusionModule


class Test_genName(unittest.TestCase):

    def test_late(self):
        self.config = {"fusionType": "LateFusion",
                       "fusionMethod": "chicken_is_heaven",
                       "classifiersNames": ["cheese", "is", "no", "disease"]}
        res = FusionModule.genName(self.config)
        self.assertEqual(res, "Late-chic")
