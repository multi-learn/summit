# from .. import metrics
#
# from ..utils.base import get_metric
#
# # Author-Info
# __author__ = "Baptiste Bauvin"
# __status__ = "Prototype"  # Production, Development, Prototype
#
#
# def print_metric_score(metric_scores, metric_list):
#     """
#     this function print the metrics scores
#
#     Parameters
#     ----------
#     metric_scores : the score of metrics
#
#     metric_list : list of metrics
#
#     Returns
#     -------
#     metric_score_string string constaining all metric results
#     """
#     metric_score_string = "\n\n"
#     for metric in metric_list:
#         metric_module = getattr(metrics, metric[0])
#         if metric[1] is not None:
#             metric_kwargs = dict(
#                 (index, metricConfig) for index, metricConfig in
#                 enumerate(metric[1]))
#         else:
#             metric_kwargs = {}
#         metric_score_string += "\tFor " + metric_module.get_config(
#             **metric_kwargs) + " : "
#         metric_score_string += "\n\t\t- Score on train : " + str(
#             metric_scores[metric[0]][0])
#         metric_score_string += "\n\t\t- Score on test : " + str(
#             metric_scores[metric[0]][1])
#         metric_score_string += "\n\n"
#     return metric_score_string
#
#
# def get_total_metric_scores(metric, train_labels, test_labels,
#                             validation_indices,
#                             learning_indices, labels):
#     """
#
#     Parameters
#     ----------
#
#     metric :
#
#     train_labels : labels of train
#
#     test_labels :  labels of test
#
#     validation_indices :
#
#     learning_indices :
#
#     labels :
#
#     Returns
#     -------
#     list of [train_score, test_score]
#     """
#     metric_module = getattr(metrics, metric[0])
#     if metric[1] is not None:
#         metric_kwargs = dict((index, metricConfig) for index, metricConfig in
#                              enumerate(metric[1]))
#     else:
#         metric_kwargs = {}
#     train_score = metric_module.score(labels[learning_indices], train_labels,
#                                       **metric_kwargs)
#     test_score = metric_module.score(labels[validation_indices], test_labels,
#                                      **metric_kwargs)
#     return [train_score, test_score]
#
#
# def get_metrics_scores(metrics, train_labels, test_labels,
#                        validation_indices, learning_indices, labels):
#     metrics_scores = {}
#     for metric in metrics:
#         metrics_scores[metric[0]] = get_total_metric_scores(metric,
#                                                             train_labels,
#                                                             test_labels,
#                                                             validation_indices,
#                                                             learning_indices,
#                                                             labels)
#     return metrics_scores
#
#
# def execute(classifier, pred_train_labels, pred_test_labels,
#             classification_indices, labels_dictionary, views, name, k_folds,
#             metrics_list, labels, directory):
#     """
#
#     Parameters
#     ----------
#     classifier : classifier used
#
#     pred_train_labels : labels of train
#
#     pred_test_labels : labels of test
#
#     classification_indices
#
#     labels_dictionary
#
#     views
#
#     name
#
#     k_folds
#
#     metrics_list
#
#     labels
#
#     Returns
#     -------
#     return tuple of (string_analysis, images_analysis, metricsScore)
#     """
#     classifier_name = classifier.short_name
#     learning_indices, validation_indices = classification_indices
#     metric_module, metric_kwargs = get_metric(metrics_list)
#     score_on_train = metric_module.score(labels[learning_indices],
#                                          pred_train_labels,
#                                          **metric_kwargs)
#     score_on_test = metric_module.score(labels[validation_indices],
#                                         pred_test_labels, **metric_kwargs)
#
#     string_analysis = "\t\tResult for multiview classification with " + classifier_name + \
#                       "\n\n" + metrics_list[0][0] + " :\n\t-On Train : " + str(
#         score_on_train) + "\n\t-On Test : " + str(
#         score_on_test) + \
#                       "\n\nDataset info :\n\t-Database name : " + name + "\n\t-Labels : " + \
#                       ', '.join(
#                           labels_dictionary.values()) + "\n\t-Views : " + ', '.join(
#         views) + "\n\t-" + str(
#         k_folds.n_splits) + \
#                       " folds\n\nClassification configuration : \n\t-Algorithm used : " + classifier_name + " with : " + classifier.get_config()
#
#     metrics_scores = get_metrics_scores(metrics_list, pred_train_labels,
#                                         pred_test_labels,
#                                         validation_indices, learning_indices,
#                                         labels)
#     string_analysis += print_metric_score(metrics_scores, metrics_list)
#     string_analysis += "\n\n Interpretation : \n\n" + classifier.get_interpretation(
#         directory, labels)
#     images_analysis = {}
#     return string_analysis, images_analysis, metrics_scores
