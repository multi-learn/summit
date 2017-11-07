import unittest
import numpy as np

from ...MonoMultiViewClassifiers.Monoview import ExecClassifMonoView


class Test_initConstants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.args = {"CL_type": "test_clf"}
        cls.X = cls.random_state.randint(0,500,(10,20))
        cls.classificationIndices = [0,2,4,6,8]
        cls.labelsNames = ["test_true", "test_false"]
        cls.name = "test"
        cls.directory = "test_dir"

    def test_simple(cls):
        kwargs, \
        t_start, \
        feat, \
        CL_type, \
        X, \
        learningRate, \
        labelsString, \
        timestr, \
        outputFileName = ExecClassifMonoView.initConstants(cls.args,
                                                           cls.X,
                                                           cls.classificationIndices,
                                                           cls.labelsNames,
                                                           cls.name,
                                                           cls.directory)
        