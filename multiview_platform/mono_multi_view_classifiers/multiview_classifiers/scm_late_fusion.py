import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.six import iteritems
import itertools
from pyscm.scm import SetCoveringMachineClassifier as scm


from ..multiview_classifiers.additions.late_fusion_utils import \
    LateFusionClassifier
from ..multiview.multiview_utils import get_examples_views_indices
from ..monoview.monoview_utils import CustomRandint, CustomUniform

classifier_class_name = "SCMLateFusionClassifier"


class DecisionStumpSCMNew(BaseEstimator, ClassifierMixin):
    """docstring for SCM
    A hands on class of SCM using decision stump, built with sklearn format in order to use sklearn function on SCM like
    CV, gridsearch, and so on ..."""

    def __init__(self, model_type='conjunction', p=0.1, max_rules=10, random_state=42):
        super(DecisionStumpSCMNew, self).__init__()
        self.model_type = model_type
        self.p = p
        self.max_rules = max_rules
        self.random_state = random_state

    def fit(self, X, y):
        self.clf = scm(model_type=self.model_type, max_rules=self.max_rules, p=self.p, random_state=self.random_state)
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        return self.clf.predict(X)

    def set_params(self, **params):
        for key, value in iteritems(params):
            if key == 'p':
                self.p = value
            if key == 'model_type':
                self.model_type = value
            if key == 'max_rules':
                self.max_rules = value

    def get_stats(self):
        return {"Binary_attributes": self.clf.model_.rules}


class SCMLateFusionClassifier(LateFusionClassifier):
    def __init__(self, random_state=None, classifier_names=None,
                 classifier_configs=None, nb_cores=1,
                 p=1, max_rules=5, order=1, model_type="conjunction", weights=None):
        self.need_probas=False
        super(SCMLateFusionClassifier, self).__init__(random_state=random_state,
                                                      classifier_names=classifier_names,
                                                      classifier_configs=classifier_configs,
                                                      nb_cores=nb_cores
                                                      )
        self.scm_classifier = None
        self.p = p
        self.max_rules = max_rules
        self.order = order
        self.model_type = model_type
        self.param_names+=["model_type", "max_rules", "p", "order"]
        self.distribs+=[["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [1,2,3]]

    def fit(self, X, y, train_indices=None, view_indices=None):
        super(SCMLateFusionClassifier, self).fit(X, y,
                                                 train_indices=train_indices,
                                                 view_indices=view_indices)
        self.scm_fusion_fit(X, y, train_indices=train_indices, view_indices=view_indices)
        return self

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        monoview_decisions = np.zeros((len(example_indices), X.nb_view),
                                      dtype=int)
        for index, view_index in enumerate(view_indices):
            monoview_decision = self.monoview_estimators[index].predict(
                X.get_v(view_index, example_indices))
            monoview_decisions[:, index] = monoview_decision
        features = self.generate_interactions(monoview_decisions)
        predicted_labels = self.scm_classifier.predict(features)
        return predicted_labels

    def scm_fusion_fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X, train_indices, view_indices)

        self.scm_classifier = DecisionStumpSCMNew(p=self.p, max_rules=self.max_rules, model_type=self.model_type,
                                                  random_state=self.random_state)
        monoview_decisions = np.zeros((len(train_indices), X.nb_view), dtype=int)
        for index, view_index in enumerate(view_indices):
            monoview_decisions[:, index] = self.monoview_estimators[index].predict(
                X.get_v(view_index, train_indices))
        features = self.generate_interactions(monoview_decisions)
        features = np.array([np.array([feat for feat in feature])
                             for feature in features])
        self.scm_classifier.fit(features, y[train_indices].astype(int))

    def generate_interactions(self, monoview_decisions):
        if self.order is None:
            self.order = monoview_decisions.shape[1]
        if self.order == 1:
            return monoview_decisions
        else:
            genrated_intercations = [monoview_decisions[:, i]
                                     for i in range(monoview_decisions.shape[1])]
            for order_index in range(self.order - 1):
                combins = itertools.combinations(range(monoview_decisions.shape[1]),
                                                 order_index + 2)
                for combin in combins:
                    generated_decision = monoview_decisions[:, combin[0]]
                    for index in range(len(combin) - 1):
                        if self.model_type == "disjunction":
                            generated_decision = np.logical_and(generated_decision,
                                                               monoview_decisions[:, combin[index + 1]])
                        else:
                            generated_decision = np.logical_or(generated_decision,
                                                              monoview_decisions[:, combin[index + 1]])
                    genrated_intercations.append(generated_decision)
            return np.transpose(np.array(genrated_intercations))

