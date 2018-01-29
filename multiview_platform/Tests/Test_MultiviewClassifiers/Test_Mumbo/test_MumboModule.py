import unittest

from ....MonoMultiViewClassifiers.MultiviewClassifiers.Mumbo import MumboModule


class Test_genName(unittest.TestCase):

    def test_simple(self):
        res = MumboModule.genName("empty")
        self.assertEqual(res, "Mumbo")

