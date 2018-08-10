class MultiviewResult(object):
    def __init__(self, classifier_name, classifier_config,
                 metrics_scores, full_labels, test_labels_multiclass):
        self.classifier_name = classifier_name
        self.classifier_config = classifier_config
        self.metrics_scores = metrics_scores
        self.full_labels = full_labels
        self.test_labels_multiclass = test_labels_multiclass