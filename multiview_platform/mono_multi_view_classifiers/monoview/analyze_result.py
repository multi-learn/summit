# from datetime import timedelta as hms
#
# from .. import metrics
# from ..utils.base import get_metric
#
#
# def get_db_config_string(name, feat, classification_indices, shape,
#                          class_labels_names, k_folds):
#     """
#
#     Parameters
#     ----------
#     name
#     feat
#     classification_indices
#     shape
#     class_labels_names
#     k_folds
#
#     Returns
#     -------
#
#     """
#     learning_rate = float(len(classification_indices[0])) / (
#             len(classification_indices[0]) + len(classification_indices[1]))
#     db_config_string = "Database configuration : \n"
#     db_config_string += "\t- Database name : " + name + "\n"
#     db_config_string += "\t- View name : " + feat + "\t View shape : " + str(
#         shape) + "\n"
#     db_config_string += "\t- Learning Rate : " + str(learning_rate) + "\n"
#     db_config_string += "\t- Labels used : " + ", ".join(
#         class_labels_names) + "\n"
#     db_config_string += "\t- Number of cross validation folds : " + str(
#         k_folds.n_splits) + "\n\n"
#     return db_config_string
#
#
# def get_classifier_config_string(grid_search, nb_cores, n_iter, cl_kwargs,
#                                  classifier,
#                                  output_file_name, y_test):
#     classifier_config_string = "Classifier configuration : \n"
#     classifier_config_string += "\t- " + classifier.get_config()[5:] + "\n"
#     classifier_config_string += "\t- Executed on " + str(
#         nb_cores) + " core(s) \n"
#     if grid_search:
#         classifier_config_string += "\t- Got configuration using randomized search with " + str(
#             n_iter) + " iterations \n"
#     classifier_config_string += "\n\n"
#     classifier_interpret_string = classifier.get_interpretation(
#         output_file_name,
#         y_test)
#     return classifier_config_string, classifier_interpret_string
#
#
# def get_metric_score(metric, y_train, y_train_pred, y_test, y_test_pred):
#     metric_module = getattr(metrics, metric[0])
#     if metric[1] is not None:
#         metric_kwargs = dict((index, metricConfig) for index, metricConfig in
#                              enumerate(metric[1]))
#     else:
#         metric_kwargs = {}
#     metric_score_train = metric_module.score(y_train, y_train_pred)
#     metric_score_test = metric_module.score(y_test, y_test_pred)
#     metric_score_string = "\tFor " + metric_module.get_config(
#         **metric_kwargs) + " : "
#     metric_score_string += "\n\t\t- Score on train : " + str(metric_score_train)
#     metric_score_string += "\n\t\t- Score on test : " + str(metric_score_test)
#     metric_score_string += "\n"
#     return metric_score_string, [metric_score_train, metric_score_test]
#
#
# def execute(name, learning_rate, k_folds, nb_cores, grid_search, metrics_list,
#             n_iter,
#             feat, cl_type, cl_kwargs, class_labels_names,
#             shape, y_train, y_train_pred, y_test, y_test_pred, time,
#             random_state, classifier, output_file_name):
#     metric_module, metric_kwargs = get_metric(metrics_list)
#     train_score = metric_module.score(y_train, y_train_pred)
#     test_score = metric_module.score(y_test, y_test_pred)
#     string_analysis = "Classification on " + name + " database for " + feat + " with " + cl_type + ".\n\n"
#     string_analysis += metrics_list[0][0] + " on train : " + str(
#         train_score) + "\n" + \
#                        metrics_list[0][0] + " on test : " + str(
#         test_score) + "\n\n"
#     string_analysis += get_db_config_string(name, feat, learning_rate, shape,
#                                             class_labels_names, k_folds)
#     classifier_config_string, classifier_intepret_string = get_classifier_config_string(
#         grid_search, nb_cores, n_iter, cl_kwargs, classifier, output_file_name,
#         y_test)
#     string_analysis += classifier_config_string
#     metrics_scores = {}
#     for metric in metrics_list:
#         metric_string, metric_score = get_metric_score(metric, y_train,
#                                                        y_train_pred, y_test,
#                                                        y_test_pred)
#         string_analysis += metric_string
#         metrics_scores[metric[0]] = metric_score
#     string_analysis += "\n\n Classification took " + str(hms(seconds=int(time)))
#     string_analysis += "\n\n Classifier Interpretation : \n"
#     string_analysis += classifier_intepret_string
#
#     image_analysis = {}
#     return string_analysis, image_analysis, metrics_scores
