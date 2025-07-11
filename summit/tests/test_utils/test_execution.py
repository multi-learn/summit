# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#
#
import os
import unittest

import numpy as np

from summit.tests.utils import rm_tmp, tmp_path, test_dataset

from summit.multiview_platform.utils import execution


class Test_parseTheArgs(unittest.TestCase):

    def setUp(self):
        self.args = []

    def test_empty_args(self):
        args = execution.parse_the_args([])


class Test_init_log_file(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        res_dir = execution.init_log_file(name="test_dataset",
                                          views=["V1", "V2", "V3"],
                                          cl_type="",
                                          log=True,
                                          debug=False,
                                          label="No",
                                          result_directory=tmp_path,
                                          args={})
        self.assertTrue(
            res_dir.startswith(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.realpath(__file__))),
                    "tmp_tests",
                    "test_dataset",
                    "started")))

    def test_no_log(self):
        res_dir = execution.init_log_file(name="test_dataset",
                                          views=["V1", "V2", "V3"],
                                          cl_type="",
                                          log=False,
                                          debug=False,
                                          label="No1",
                                          result_directory=tmp_path,
                                          args={})
        self.assertTrue(res_dir.startswith(os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "tmp_tests", "test_dataset", "started")))

    def test_debug(self):
        res_dir = execution.init_log_file(name="test_dataset",
                                          views=["V1", "V2", "V3"],
                                          cl_type="",
                                          log=True,
                                          debug=True,
                                          label="No",
                                          result_directory=tmp_path,
                                          args={})
        self.assertTrue(res_dir.startswith(os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "tmp_tests", "test_dataset", "debug_started")))


class Test_gen_k_folds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.statsIter = 1

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        folds_list = execution.gen_k_folds(stats_iter=1,
                                           nb_folds=4,
                                           stats_iter_random_states=np.random.RandomState(42))
        self.assertEqual(folds_list[0].n_splits, 4)
        self.assertEqual(len(folds_list), 1)

    def test_multple_iters(self):
        folds_list = execution.gen_k_folds(stats_iter=2,
                                           nb_folds=4,
                                           stats_iter_random_states=[np.random.RandomState(42), np.random.RandomState(43)])
        self.assertEqual(folds_list[0].n_splits, 4)
        self.assertEqual(len(folds_list), 2)

    def test_list_rs(self):
        folds_list = execution.gen_k_folds(stats_iter=1,
                                           nb_folds=4,
                                           stats_iter_random_states=[np.random.RandomState(42)])
        self.assertEqual(folds_list[0].n_splits, 4)
        self.assertEqual(len(folds_list), 1)


class Test_init_views(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.statsIter = 1

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        views, views_indices, all_views = execution.init_views(
            test_dataset, ["ViewN1", "ViewN2"])
        self.assertEqual(views, ["ViewN1", "ViewN2"])
        self.assertEqual(views_indices, [1, 2])
        self.assertEqual(all_views, ["ViewN0", "ViewN1", "ViewN2"])

        views, views_indices, all_views = execution.init_views(
            test_dataset, None)
        self.assertEqual(views, ["ViewN0", "ViewN1", "ViewN2"])
        self.assertEqual(views_indices, range(3))
        self.assertEqual(all_views, ["ViewN0", "ViewN1", "ViewN2"])


class Test_find_dataset_names(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        os.mkdir(tmp_path)
        with open(os.path.join(tmp_path, "test.txt"), "w") as file_stream:
            file_stream.write("test")
        with open(os.path.join(tmp_path, "test1.txt"), "w") as file_stream:
            file_stream.write("test")

    @classmethod
    def tearDownClass(cls):
        rm_tmp()

    def test_simple(self):
        path, names = execution.find_dataset_names(tmp_path, ".txt", ["test"])
        self.assertEqual(path, tmp_path)
        self.assertEqual(names, ["test"])
        path, names = execution.find_dataset_names(
            tmp_path, ".txt", ["test", 'test1'])
        self.assertEqual(path, tmp_path)
        self.assertIn("test1", names)
        path, names = execution.find_dataset_names(
            os.path.join("examples","data"), ".hdf5", ["all"])
        self.assertIn("doc_summit", names)
        self.assertRaises(ValueError, execution.find_dataset_names, tmp_path + "test", ".txt",
                          ["test"])
        self.assertRaises(
            ValueError,
            execution.find_dataset_names,
            tmp_path,
            ".txt",
            ["ah"])


class Test_initStatsIterRandomStates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.statsIter = 1

    def test_one_statiter(cls):
        cls.state = cls.random_state.get_state()[1]
        statsIterRandomStates = execution.init_stats_iter_random_states(

            cls.statsIter, cls.random_state)
        np.testing.assert_array_equal(statsIterRandomStates[0].get_state()[1],
                                      cls.state)

    def test_multiple_iter(cls):
        cls.statsIter = 3
        statsIterRandomStates = execution.init_stats_iter_random_states(

            cls.statsIter, cls.random_state)
        cls.assertAlmostEqual(len(statsIterRandomStates), 3)
        cls.assertNotEqual(statsIterRandomStates[0].randint(5000),
                           statsIterRandomStates[1].randint(5000))
        cls.assertNotEqual(statsIterRandomStates[0].randint(5000),
                           statsIterRandomStates[2].randint(5000))
        cls.assertNotEqual(statsIterRandomStates[2].randint(5000),
                           statsIterRandomStates[1].randint(5000))


class Test_getDatabaseFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.name = "zrtTap"
        cls.type = ".csv"

    def test_simple(cls):
        getDB = execution.get_database_function(cls.name, cls.type)
        from summit.multiview_platform.utils.get_multiview_db import \
            get_classic_db_csv
        cls.assertEqual(getDB, get_classic_db_csv)

    def test_hdf5(cls):
        cls.type = ".hdf5"
        getDB = execution.get_database_function(cls.name, cls.type)
        from summit.multiview_platform.utils.get_multiview_db import \
            get_classic_db_hdf5
        cls.assertEqual(getDB, get_classic_db_hdf5)

    def test_plausible_hdf5(cls):
        cls.name = "plausible"
        cls.type = ".hdf5"
        getDB = execution.get_database_function(cls.name, cls.type)
        from summit.multiview_platform.utils.get_multiview_db import \
            get_plausible_db_hdf5
        cls.assertEqual(getDB, get_plausible_db_hdf5)


class Test_initRandomState(unittest.TestCase):

    def setUp(self):
        rm_tmp()
        os.mkdir(tmp_path)

    def tearDown(self):
        os.rmdir(tmp_path)

    def test_random_state_42(self):
        randomState_42 = np.random.RandomState(42)
        randomState = execution.init_random_state("42",
                                                  tmp_path)
        os.remove(tmp_path + "random_state.pickle")
        np.testing.assert_array_equal(randomState.beta(1, 100, 100),
                                      randomState_42.beta(1, 100, 100))

    def test_random_state_pickle(self):
        randomState_to_pickle = execution.init_random_state(None,
                                                            tmp_path)
        pickled_randomState = execution.init_random_state(
            tmp_path + "random_state.pickle",
            tmp_path)
        os.remove(tmp_path + "random_state.pickle")

        np.testing.assert_array_equal(randomState_to_pickle.beta(1, 100, 100),
                                      pickled_randomState.beta(1, 100, 100))


class FakeArg():

    def __init__(self):
        self.name = "zrtTap"
        self.CL_type = ["fromage", "jambon"]
        self.views = ["view1", "view2"]
        self.log = True


# Impossible to test as the main directory is notthe same for the exec and the test
# class Test_initLogFile(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.fakeArgs = FakeArg()
#         cls.timestr = time.strftime("%Y_%m_%d-%H_%M")
#
#     def test_initLogFile(cls):
#         cls.timestr = time.strftime("%Y_%m_%d-%H_%M")
#         execution.initLogFile(cls.fakeArgs)
#         cls.assertIn("zrtTap", os.listdir("mutliview_platform/results"), "Database directory not created")
#         cls.assertIn("started_"+cls.timestr, os.listdir("mutliview_platform/results/zrtTap"),"experimentation dir not created")
#         cls.assertIn(cls.timestr + "-" + ''.join(cls.fakeArgs.CL_type) + "-" + "_".join(
#         cls.fakeArgs.views) + "-" + cls.fakeArgs.name + "-LOG.log", os.listdir("mutliview_platform/results/zrtTap/"+"started_"+cls.timestr), "logfile was not created")
#
#     @classmethod
#     def tearDownClass(cls):
#         shutil.rmtree("summit/results/zrtTap")
#         pass


class Test_genSplits(unittest.TestCase):

    def setUp(self):
        self.stastIter = 3
        self.statsIterRandomStates = [np.random.RandomState(42 + i + 1) for i in
                                      range(self.stastIter)]
        self.random_state = np.random.RandomState(42)
        self.X_indices = self.random_state.randint(0, 500, 50)
        self.labels = np.zeros(500)
        self.labels[self.X_indices[:10]] = 1
        self.labels[self.X_indices[11:30]] = 2  # To test multiclass
        self.splitRatio = 0.2

    def test_simple(self):
        splits = execution.gen_splits(self.labels, self.splitRatio,
                                      self.statsIterRandomStates)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[1]), 2)
        self.assertEqual(type(splits[1][0]), np.ndarray)
        self.assertAlmostEqual(len(splits[1][0]), 0.8 * 500)
        self.assertAlmostEqual(len(splits[1][1]), 0.2 * 500)
        self.assertGreater(len(np.where(self.labels[splits[1][0]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][0]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][0]] == 2)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[1][1]] == 2)[0]), 0)

    def test_genSplits_no_iter(self):
        splits = execution.gen_splits(self.labels, self.splitRatio,
                                      self.statsIterRandomStates)
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(type(splits[0][0]), np.ndarray)
        self.assertAlmostEqual(len(splits[0][0]), 0.8 * 500)
        self.assertAlmostEqual(len(splits[0][1]), 0.2 * 500)
        self.assertGreater(len(np.where(self.labels[splits[0][0]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][0]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][0]] == 2)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]] == 0)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]] == 1)[0]), 0)
        self.assertGreater(len(np.where(self.labels[splits[0][1]] == 2)[0]), 0)


class Test_genKFolds(unittest.TestCase):

    def setUp(self):
        self.statsIter = 2
        self.nbFolds = 5
        self.statsIterRandomStates = [np.random.RandomState(42),
                                      np.random.RandomState(94)]

    def test_genKFolds_iter(self):
        pass


class Test_genDirecortiesNames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.directory = tmp_path
        cls.stats_iter = 5

    def test_simple_ovo(cls):
        directories = execution.gen_direcorties_names(cls.directory,
                                                      cls.stats_iter)
        cls.assertEqual(len(directories), 5)
        cls.assertEqual(directories[0], os.path.join(tmp_path, "iter_1"))
        cls.assertEqual(directories[-1], os.path.join(tmp_path, "iter_5"))

    def test_ovo_no_iter(cls):
        cls.stats_iter = 1
        directories = execution.gen_direcorties_names(cls.directory,
                                                      cls.stats_iter)
        cls.assertEqual(len(directories), 1)
        cls.assertEqual(directories[0], tmp_path)


class Test_genArgumentDictionaries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.labelsDictionary = {0: "yes", 1: "No", 2: "Maybe"}
        cls.direcories = ["Res/iter_1", "Res/iter_2"]
        cls.multiclassLabels = [np.array([0, 1, -100, 1, 0]),
                                np.array([1, 0, -100, 1, 0]),
                                np.array([0, 1, -100, 0, 1])]
        cls.labelsCombinations = [[0, 1], [0, 2], [1, 2]]
        cls.indicesMulticlass = [[[[], []], [[], []], [[], []]], [[], [], []]]


if __name__ == '__main__':
    unittest.main()
