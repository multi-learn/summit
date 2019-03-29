
from ..MonoviewUtils import change_label_to_minus
from .BoostUtils import StumpsClassifiersGenerator, BaseBoost

class PregenClassifier(BaseBoost):

    def pregen_voters(self, X, y=None):
        if y is not None:
            neg_y = change_label_to_minus(y)
            if self.estimators_generator is None:
                self.estimators_generator = StumpsClassifiersGenerator(
                    n_stumps_per_attribute=self.n_stumps,
                    self_complemented=self.self_complemented)
            self.estimators_generator.fit(X, neg_y)
        else:
            neg_y=None
        classification_matrix = self._binary_classification_matrix(X)
        return classification_matrix, neg_y
