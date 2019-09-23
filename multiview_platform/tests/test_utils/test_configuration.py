import os
import unittest

import numpy as np

from multiview_platform.MonoMultiViewClassifiers.utils import configuration

class Test_get_the_args(unittest.TestCase):

    def setUp(self):
        self.path_to_config_file = "multiview_platform/Tests/tmp_tests/config_temp.ini"
        os.mkdir("multiview_platform/Tests/tmp_tests")
        config_file = open(self.path_to_config_file, "w")
        config_file.write("[Base]\nfirst_arg = int ; 10\nsecond_arg = list_float ; 12.5 1e-06\n[Classification]\nthird_arg = bool ; yes")
        config_file.close()

    def tearDown(self):
        os.remove("multiview_platform/Tests/tmp_tests/config_temp.ini")
        os.rmdir("multiview_platform/Tests/tmp_tests")

    def test_file_loading(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertEqual(type(config_dict), dict)

    def test_dict_format(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertIn("Base", config_dict)
        self.assertIn("Classification", config_dict)
        self.assertIn("first_arg", config_dict["Base"])
        self.assertIn("third_arg", config_dict["Classification"])

    def test_arguments(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertEqual(config_dict["Base"]["first_arg"], 10)
        self.assertEqual(config_dict["Base"]["second_arg"], [12.5, 1e-06])
        self.assertEqual(config_dict["Classification"]["third_arg"], True)

class Test_format_the_args(unittest.TestCase):

    def test_bool(self):
        value = configuration.format_raw_arg("bool ; yes")
        self.assertEqual(value, True)

    def test_int(self):
        value = configuration.format_raw_arg("int ; 1")
        self.assertEqual(value, 1)

    def test_float(self):
        value = configuration.format_raw_arg("float ; 1.5")
        self.assertEqual(value, 1.5)

    def test_string(self):
        value = configuration.format_raw_arg("str ; chicken_is_heaven")
        self.assertEqual(value, "chicken_is_heaven")

    def test_list_bool(self):
        value = configuration.format_raw_arg("list_bool ; yes no yes yes")
        self.assertEqual(value, [True, False, True, True])

    def test_list_int(self):
        value = configuration.format_raw_arg("list_int ; 1 2 3 4")
        self.assertEqual(value, [1,2,3,4])

    def test_list_float(self):
        value = configuration.format_raw_arg("list_float ; 1.5 1.6 1.7")
        self.assertEqual(value, [1.5, 1.6, 1.7])

    def test_list_string(self):
        value = configuration.format_raw_arg("list_str ; list string")
        self.assertEqual(value, ["list", "string"])

