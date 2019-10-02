import inspect


from ...multiview.multiview_utils import BaseMultiviewClassifier, get_monoview_classifier


class BaseLateFusionClassifier(BaseMultiviewClassifier):

    def init_monoview_estimator(self, classifier_name, classifier_index=None):
        if classifier_index is not None:
            classifier_configs = self.classifier_configs[classifier_index]
        else:
            classifier_configs = self.classifier_configs
        if classifier_configs is not None and classifier_name in classifier_configs:
            if 'random_state' in inspect.getfullargspec(
                    get_monoview_classifier(classifier_name).__init__).args:
                estimator = get_monoview_classifier(classifier_name)(
                    random_state=self.random_state,
                    **classifier_configs[classifier_name])
            else:
                estimator = get_monoview_classifier(classifier_name)(
                    **classifier_configs[classifier_name])
        else:
            if 'random_state' in inspect.getfullargspec(
                    get_monoview_classifier(classifier_name).__init__).args:
                estimator = get_monoview_classifier(classifier_name)(
                    random_state=self.random_state)
            else:
                estimator = get_monoview_classifier(classifier_name)()
        return estimator
