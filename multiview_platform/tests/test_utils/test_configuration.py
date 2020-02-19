import os
import unittest
import yaml
import numpy as np

from multiview_platform.tests.utils import rm_tmp, tmp_path
from multiview_platform.mono_multi_view_classifiers.utils import configuration


class Test_get_the_args(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rm_tmp()
        cls.path_to_config_file = tmp_path+"config_temp.yml"
        path_file = os.path.dirname(os.path.abspath(__file__))
        make_tmp_dir = os.path.join(path_file, "../tmp_tests")
        os.mkdir(make_tmp_dir)
        data = {"log": 10, "name":[12.5, 1e-06], "type":True}
        with open(cls.path_to_config_file, "w") as config_file:
            yaml.dump(data, config_file)

    @classmethod
    def tearDownClass(cls):
        os.remove(tmp_path+"config_temp.yml")
        os.rmdir(tmp_path)

    def test_file_loading(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertEqual(type(config_dict), dict)

    def test_dict_format(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertIn("log", config_dict)
        self.assertIn("name", config_dict)

    def test_arguments(self):
        config_dict = configuration.get_the_args(self.path_to_config_file)
        self.assertEqual(config_dict["log"], 10)
        self.assertEqual(config_dict["name"], [12.5, 1e-06])
        self.assertEqual(config_dict["type"], True)

# class Test_format_the_args(unittest.TestCase):
#
#     def test_bool(self):
#         value = configuration.format_raw_arg("bool ; yes")
#         self.assertEqual(value, True)
#
#     def test_int(self):
#         value = configuration.format_raw_arg("int ; 1")
#         self.assertEqual(value, 1)
#
#     def test_float(self):
#         value = configuration.format_raw_arg("float ; 1.5")
#         self.assertEqual(value, 1.5)
#
#     def test_string(self):
#         value = configuration.format_raw_arg("str ; chicken_is_heaven")
#         self.assertEqual(value, "chicken_is_heaven")
#
#     def test_list_bool(self):
#         value = configuration.format_raw_arg("list_bool ; yes no yes yes")
#         self.assertEqual(value, [True, False, True, True])
#
#     def test_list_int(self):
#         value = configuration.format_raw_arg("list_int ; 1 2 3 4")
#         self.assertEqual(value, [1,2,3,4])
#
#     def test_list_float(self):
#         value = configuration.format_raw_arg("list_float ; 1.5 1.6 1.7")
#         self.assertEqual(value, [1.5, 1.6, 1.7])
#
#     def test_list_string(self):
#         value = configuration.format_raw_arg("list_str ; list string")
#         self.assertEqual(value, ["list", "string"])

if __name__ == '__main__':
    unittest.main()