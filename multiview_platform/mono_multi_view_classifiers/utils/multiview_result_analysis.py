from .. import metrics

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def print_metric_score(metric_scores, metrics):
    metric_score_string = "\n\n"
    for metric in metrics:
        metric_module = getattr(metrics, metric[0])
        if metric[1] is not None:
            metric_kwargs = dict((index, metricConfig) for index, metricConfig in
                                 enumerate(metric[1]))
        else:
            metric_kwargs = {}
        metric_score_string += "\tFor " + metric_module.get_config(
            **metric_kwargs) + " : "
        metric_score_string += "\n\t\t- Score on train : " + str(
            metric_scores[metric[0]][0])
        metric_score_string += "\n\t\t- Score on test : " + str(
            metric_scores[metric[0]][1])
        metric_score_string += "\n\n"
    return metric_score_string


def get_total_metric_scores(metric, train_labels, test_labels, validation_indices,
                            learning_indices, labels):
    metric_module = getattr(metrics, metric[0])
    if metric[1] is not None:
        metric_kwargs = dict((index, metricConfig) for index, metricConfig in
                             enumerate(metric[1]))
    else:
        metric_kwargs = {}
    train_score = metric_module.score(labels[learning_indices], train_labels,
                                      **metric_kwargs)
    test_score = metric_module.score(labels[validation_indices], test_labels,
                                     **metric_kwargs)
    return [train_score, test_score]


def get_metrics_scores(metrics_var, train_labels, test_labels,
                       validation_indices, learning_indices, labels):
    metrics_scores = {}
    for metric in metrics_var:
        metrics_scores[metric[0]] = get_total_metric_scores(metric, train_labels,
                                                            test_labels,
                                                            validation_indices,
                                                            learning_indices, labels)
    return metrics_scores
