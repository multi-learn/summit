import inspect


from ...multiview.multiview_utils import get_monoview_classifier


class BaseFusionClassifier():

    def init_monoview_estimator(self, classifier_name, classifier_config,
                                classifier_index=None,):
        if classifier_index is not None:
            if classifier_config is not None:
                classifier_configs = classifier_config[classifier_name]
            else:
                classifier_configs = None
        else:
            classifier_configs = classifier_config
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
