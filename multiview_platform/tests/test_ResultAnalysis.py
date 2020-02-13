import unittest
import numpy as np
import pandas as pd
import os

from multiview_platform.mono_multi_view_classifiers import result_analysis
from multiview_platform.mono_multi_view_classifiers.multiview.multiview_utils import MultiviewResult
from multiview_platform.mono_multi_view_classifiers.monoview.monoview_utils import MonoviewResult


class Test_get_arguments(unittest.TestCase):

    def setUp(self):
        self.benchamrk_argument_dictionaries = [{"flag":"good_flag", "valid":True},
                                                {"flag":"bad_flag", "valid":False}]

    def test_benchmark_wanted(self):
        argument_dict = result_analysis.get_arguments(self.benchamrk_argument_dictionaries, "good_flag")
        self.assertTrue(argument_dict["valid"])


class Test_get_metrics_scores_biclass(unittest.TestCase):


    def test_simple(self):
        metrics = [["accuracy_score"], ["f1_score"]]
        results = [MonoviewResult(0,
                                  "ada",
                                  "0",
                                  {"accuracy_score":[0.9, 0.95],
                                   "f1_score":[0.91, 0.96]}
                                  , "", "", "", "", "",)]
        metrics_scores = result_analysis.get_metrics_scores_biclass(metrics,
                                                                    results)
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score"], pd.DataFrame)
        np.testing.assert_array_equal(np.array(metrics_scores["accuracy_score"].loc["train"]), np.array([0.9]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score"].loc["test"]),
            np.array([0.95]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["train"]),
            np.array([0.91]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["test"]),
            np.array([0.96]))
        np.testing.assert_array_equal(np.array(metrics_scores["f1_score"].columns),
                                      np.array(["ada-0"]))

    def test_multiple_monoview_classifiers(self):
        metrics = [["accuracy_score"], ["f1_score"]]
        results = [MonoviewResult(view_index=0,
                                  classifier_name="ada",
                                  view_name="0",
                                  metrics_scores={"accuracy_score": [0.9, 0.95],
                                   "f1_score": [0.91, 0.96]},
                                  full_labels_pred="",
                                  classifier_config="",
                                  test_folds_preds="",
                                  classifier="",
                                  n_features=""),
                   MonoviewResult(view_index=0,
                                  classifier_name="dt",
                                  view_name="1",
                                  metrics_scores={"accuracy_score": [0.8, 0.85],
                                   "f1_score": [0.81, 0.86]},
                                  full_labels_pred="",
                                  classifier_config="",
                                  test_folds_preds="",
                                  classifier="",
                                  n_features="")
                   ]
        metrics_scores = result_analysis.get_metrics_scores_biclass(metrics,
                                                                    results)
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score"], pd.DataFrame)
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score"].loc["train"]),
            np.array([0.9, 0.8]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score"].loc["test"]),
            np.array([0.95, 0.85]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["train"]),
            np.array([0.91, 0.81]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["test"]),
            np.array([0.96, 0.86]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].columns),
            np.array(["ada-0", "dt-1"]))

    def test_mutiview_result(self):
        metrics = [["accuracy_score"], ["f1_score"]]
        results = [MultiviewResult("mv", "", {"accuracy_score": [0.7, 0.75],
                                   "f1_score": [0.71, 0.76]}, "", ),
                   MonoviewResult(view_index=0,
                                  classifier_name="dt",
                                  view_name="1",
                                  metrics_scores={"accuracy_score": [0.8, 0.85],
                                                  "f1_score": [0.81, 0.86]},
                                  full_labels_pred="",
                                  classifier_config="",
                                  test_folds_preds="",
                                  classifier="",
                                  n_features="")
                   ]
        metrics_scores = result_analysis.get_metrics_scores_biclass(metrics,
                                                                    results)
        self.assertIsInstance(metrics_scores, dict)
        self.assertIsInstance(metrics_scores["accuracy_score"], pd.DataFrame)
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score"].loc["train"]),
            np.array([0.7, 0.8]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["accuracy_score"].loc["test"]),
            np.array([0.75, 0.85]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["train"]),
            np.array([0.71, 0.81]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].loc["test"]),
            np.array([0.76, 0.86]))
        np.testing.assert_array_equal(
            np.array(metrics_scores["f1_score"].columns),
            np.array(["mv", "dt-1"]))

class Test_get_example_errors_biclass(unittest.TestCase):

    def test_simple(self):
        ground_truth = np.array([0,1,0,1,0,1,0,1, -100])
        results = [MultiviewResult("mv", "", {"accuracy_score": [0.7, 0.75],
                                              "f1_score": [0.71, 0.76]},
                                   np.array([0,0,0,0,1,1,1,1,1]),
                                   ),
                   MonoviewResult(0,
                                  "dt",
                                  "1",
                                  {"accuracy_score": [0.8, 0.85],
                                   "f1_score": [0.81, 0.86]}
                                  , np.array([0,0,1,1,0,0,1,1,0]), "", "",
                                  "", "",)
                   ]
        example_errors = result_analysis.get_example_errors_biclass(ground_truth,
                                                                    results)
        self.assertIsInstance(example_errors, dict)
        np.testing.assert_array_equal(example_errors["mv"],
                                      np.array([1,0,1,0,0,1,0,1,-100]))
        np.testing.assert_array_equal(example_errors["dt-1"],
                                      np.array([1, 0, 0, 1, 1, 0, 0, 1,-100]))


class Test_init_plot(unittest.TestCase):

    def test_simple(self):
        results = []
        metric_name = "acc"
        data = np.random.RandomState(42).uniform(0,1,(2,2))
        metric_dataframe = pd.DataFrame(index=["train", "test"],
                                        columns=["dt-1", "mv"], data=data)
        directory = "dir"
        database_name = 'db'
        labels_names = ['lb1', "lb2"]
        train, test, classifier_names, \
        file_name, nb_results, results = result_analysis.init_plot(results,
                                                                   metric_name,
                                                                   metric_dataframe,
                                                                   directory,
                                                                   database_name,
                                                                   labels_names)
        self.assertEqual(file_name, os.path.join("dir", "db-lb1_vs_lb2-acc"))
        np.testing.assert_array_equal(train, data[0,:])
        np.testing.assert_array_equal(test, data[1, :])
        np.testing.assert_array_equal(classifier_names, np.array(["dt-1", "mv"]))
        self.assertEqual(nb_results, 2)
        self.assertEqual(results, [["dt-1", "acc", data[1,0], 0],
                                   ["mv", "acc", data[1,1], 0]])

class Test_gen_error_data(unittest.TestCase):

    def test_simple(self):
        random_state = np.random.RandomState(42)
        ada_data = random_state.randint(0,2,size=7)
        mv_data = random_state.randint(0, 2, size=7)
        example_errors = {"ada-1": ada_data,
                          "mv": mv_data}
        nb_classifiers, nb_examples, classifiers_names, \
        data_2d, error_on_examples = result_analysis.gen_error_data(example_errors)
        self.assertEqual(nb_classifiers, 2)
        self.assertEqual(nb_examples, 7)
        self.assertEqual(classifiers_names, ["ada-1", "mv"])
        np.testing.assert_array_equal(data_2d, np.array([ada_data, mv_data]).transpose())
        np.testing.assert_array_equal(error_on_examples, -1*(ada_data+mv_data)/nb_classifiers)


class Test_format_previous_results(unittest.TestCase):

    def test_simple(self):
        biclass_results = {"metrics_scores":[], "example_errors":[], "feature_importances":[], "labels":[]}
        random_state = np.random.RandomState(42)

        # Gen metrics data
        metrics_1_data = random_state.uniform(size=(2,2))
        metrics_2_data = random_state.uniform(size=(2,2))
        metric_1_df = pd.DataFrame(data=metrics_1_data, index=["train", "test"],
                                   columns=["ada-1", "mv"])
        metric_2_df = pd.DataFrame(data=metrics_2_data, index=["train", "test"],
                                   columns=["ada-1", "mv"])
        biclass_results["metrics_scores"].append({"acc": metric_1_df})
        biclass_results["metrics_scores"].append({"acc": metric_2_df})

        # Gen error data
        ada_error_data_1 = random_state.randint(0,2,7)
        ada_error_data_2 = random_state.randint(0, 2, 7)
        ada_sum = ada_error_data_1+ada_error_data_2
        mv_error_data_1 = random_state.randint(0, 2, 7)
        mv_error_data_2 = random_state.randint(0, 2, 7)
        mv_sum = mv_error_data_1+mv_error_data_2
        biclass_results["example_errors"].append({})
        biclass_results["example_errors"].append({})
        biclass_results["example_errors"][0]["ada-1"] = ada_error_data_1
        biclass_results["example_errors"][0]["mv"] = mv_error_data_1
        biclass_results["example_errors"][1]["ada-1"] = ada_error_data_2
        biclass_results["example_errors"][1]["mv"] = mv_error_data_2

        # Running the function
        metric_analysis, error_analysis, feature_importances, feature_stds,labels = result_analysis.format_previous_results(biclass_results)
        mean_df = pd.DataFrame(data=np.mean(np.array([metrics_1_data,
                                                      metrics_2_data]),
                                            axis=0),
                               index=["train", "test"],
                               columns=["ada-1", "mvm"])
        std_df =  pd.DataFrame(data=np.std(np.array([metrics_1_data,
                                                      metrics_2_data]),
                                            axis=0),
                               index=["train", "test"],
                               columns=["ada-1", "mvm"])

        # Testing
        np.testing.assert_array_equal(metric_analysis["acc"]["mean"].loc["train"],
                                      mean_df.loc["train"])
        np.testing.assert_array_equal(metric_analysis["acc"]["mean"].loc["test"],
            mean_df.loc["test"])
        np.testing.assert_array_equal(metric_analysis["acc"]["std"].loc["train"],
            std_df.loc["train"])
        np.testing.assert_array_equal(metric_analysis["acc"]["std"].loc["test"],
            std_df.loc["test"])
        np.testing.assert_array_equal(ada_sum, error_analysis["ada-1"])
        np.testing.assert_array_equal(mv_sum, error_analysis["mv"])


class Test_gen_error_data_glob(unittest.TestCase):

    def test_simple(self):
        random_state = np.random.RandomState(42)

        ada_error_data_1 = random_state.randint(0,2,7)
        ada_error_data_2 = random_state.randint(0, 2, 7)
        ada_sum = ada_error_data_1+ada_error_data_2
        mv_error_data_1 = random_state.randint(0, 2, 7)
        mv_error_data_2 = random_state.randint(0, 2, 7)
        mv_sum = mv_error_data_1+mv_error_data_2

        combi_results = {"ada-1":ada_sum, "mv": mv_sum}

        stats_iter = 2

        nb_examples, nb_classifiers, \
        data, error_on_examples, \
        classifier_names = result_analysis.gen_error_data_glob(combi_results,
                                                              stats_iter)
        self.assertEqual(nb_examples, 7)
        self.assertEqual(nb_classifiers, 2)
        np.testing.assert_array_equal(data, np.array([ada_sum, mv_sum]).transpose())
        np.testing.assert_array_equal(error_on_examples, -1*np.sum(np.array([ada_sum, mv_sum]), axis=0)+(nb_classifiers*stats_iter))
        self.assertEqual(classifier_names, ["ada-1", "mv"])







