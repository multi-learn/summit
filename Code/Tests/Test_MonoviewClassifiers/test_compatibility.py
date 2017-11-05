import unittest
import os

from ...MonoMultiViewClassifiers import MonoviewClassifiers

class Test_methods(unittest.TestCase):

    def test_simple(self):
        for fileName in os.listdir("Code/MonoMultiViewClassifiers/MonoviewClassifiers"):
            if fileName[-3:] == ".py" and fileName != "__init__.py":
                monoview_classifier_module = getattr(MonoviewClassifiers, fileName[:-3])
                self.assertIn("canProbas", dir(monoview_classifier_module),
                              fileName[:-3]+" must have canProbas method implemented")
                self.assertIn("fit", dir(monoview_classifier_module),
                              fileName[:-3]+" must have fit method implemented")
                self.assertIn("paramsToSet", dir(monoview_classifier_module),
                              fileName[:-3]+" must have paramsToSet method implemented")
                self.assertIn("getKWARGS", dir(monoview_classifier_module),
                              fileName[:-3]+" must have getKWARGS method implemented")
                self.assertIn("randomizedSearch", dir(monoview_classifier_module),
                              fileName[:-3]+" must have randomizedSearch method implemented")
                self.assertIn("getConfig", dir(monoview_classifier_module),
                              fileName[:-3]+" must have getConfig method implemented")
                self.assertIn("getInterpret", dir(monoview_classifier_module),
                              fileName[:-3]+" must have getInterpret method implemented")

class Test_inputs(unittest.TestCase):

    def test_canProbas(self):
        for fileName in os.listdir("Code/MonoMultiViewClassifiers/MonoviewClassifiers"):
            if fileName[-3:] == ".py" and fileName != "__init__.py":
                monoview_classifier_module = getattr(MonoviewClassifiers, fileName[:-3])
                res = monoview_classifier_module.canProbas()
                self.assertEqual(type(res), bool, "canProbas must return a boolean")
                with self.assertRaises(TypeError, msg="canProbas must have 0 args") as catcher:
                    monoview_classifier_module.canProbas(35)
