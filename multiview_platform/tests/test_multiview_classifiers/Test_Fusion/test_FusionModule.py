import unittest

from ....mono_multi_view_classifiers.multiview_classifiers.fusion import \
    fusion


class Test_genName(unittest.TestCase):

    def test_late(self):
        self.config = {"fusionType": "LateFusion",
                       "fusionMethod": "chicken_is_heaven",
                       "classifiersNames": ["cheese", "is", "no", "disease"]}
        res = fusion.genName(self.config)
        self.assertEqual(res, "Late-chic")
