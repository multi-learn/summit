import unittest
import numpy as np
import pandas as pd

from multiview_platform.mono_multi_view_classifiers.monoview.monoview_utils import MonoviewResult
from multiview_platform.mono_multi_view_classifiers.multiview.multiview_utils import MultiviewResult

from multiview_platform.mono_multi_view_classifiers.result_analysis.execution import format_previous_results, get_arguments

class Test_format_previous_results(unittest.TestCase):

    def test_simple(self):
        iter_results = {"metrics_scores":[], "example_errors":[], "feature_importances":[], "labels":[], "durations":[]}
        random_state = np.random.RandomState(42)

        # Gen metrics data
        metrics_1_data = random_state.uniform(size=(2,2))
        metrics_2_data = random_state.uniform(size=(2,2))
        metric_1_df = pd.DataFrame(data=metrics_1_data, index=["train", "test"],
                                   columns=["ada-1", "mv"])
        metric_2_df = pd.DataFrame(data=metrics_2_data, index=["train", "test"],
                                   columns=["ada-1", "mv"])
        iter_results["metrics_scores"].append({"acc": metric_1_df})
        iter_results["metrics_scores"].append({"acc": metric_2_df})

        # Gen error data
        ada_error_data_1 = random_state.randint(0,2,7)
        ada_error_data_2 = random_state.randint(0, 2, 7)
        ada_sum = ada_error_data_1+ada_error_data_2
        mv_error_data_1 = random_state.randint(0, 2, 7)
        mv_error_data_2 = random_state.randint(0, 2, 7)
        mv_sum = mv_error_data_1+mv_error_data_2
        iter_results["example_errors"].append({})
        iter_results["example_errors"].append({})
        iter_results["example_errors"][0]["ada-1"] = ada_error_data_1
        iter_results["example_errors"][0]["mv"] = mv_error_data_1
        iter_results["example_errors"][1]["ada-1"] = ada_error_data_2
        iter_results["example_errors"][1]["mv"] = mv_error_data_2

        iter_results["durations"].append(pd.DataFrame(index=["ada-1", "mv"],
                                                         columns=["plif", "plaf"],
                                                         data=np.zeros((2,2))))
        iter_results["durations"].append(pd.DataFrame(index=["ada-1", "mv"],
                                                         columns=["plif",
                                                                  "plaf"],
                                                         data=np.ones((2, 2))))

        # Running the function
        metric_analysis, error_analysis, \
        feature_importances, feature_stds, \
        labels, durations_mean, duration_std = format_previous_results(iter_results)
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
        self.assertEqual(durations_mean.at["ada-1", 'plif'], 0.5)

class Test_get_arguments(unittest.TestCase):

    def setUp(self):
        self.benchamrk_argument_dictionaries = [{"flag":"good_flag", "valid":True},
                                                {"flag":"bad_flag", "valid":False}]

    def test_benchmark_wanted(self):
        argument_dict = get_arguments(self.benchamrk_argument_dictionaries, "good_flag")
        self.assertTrue(argument_dict["valid"])
