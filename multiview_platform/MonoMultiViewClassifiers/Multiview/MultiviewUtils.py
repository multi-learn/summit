from .. import MultiviewClassifiers

class MultiviewResult(object):
    def __init__(self, classifier_name, classifier_config,
                 metrics_scores, full_labels, test_labels_multiclass):
        self.classifier_name = classifier_name
        self.classifier_config = classifier_config
        self.metrics_scores = metrics_scores
        self.full_labels_pred = full_labels
        self.y_test_multiclass_pred = test_labels_multiclass

    def get_classifier_name(self):
        multiviewClassifierPackage = getattr(MultiviewClassifiers, self.classifier_name)
        multiviewClassifierModule = getattr(multiviewClassifierPackage, self.classifier_name + "Module")
        return multiviewClassifierModule.genName(self.classifier_config)