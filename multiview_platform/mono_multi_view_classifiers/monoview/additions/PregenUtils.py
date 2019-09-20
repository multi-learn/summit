from .BoostUtils import StumpsClassifiersGenerator, BaseBoost, \
    TreeClassifiersGenerator
from ..monoview_utils import change_label_to_minus


class PregenClassifier(BaseBoost):

    def pregen_voters(self, X, y=None, generator="Stumps"):
        if y is not None:
            neg_y = change_label_to_minus(y)
            if generator is "Stumps":
                self.estimators_generator = StumpsClassifiersGenerator(
                    n_stumps_per_attribute=self.n_stumps,
                    self_complemented=self.self_complemented)
            elif generator is "Trees":
                self.estimators_generator = TreeClassifiersGenerator(
                    n_trees=self.n_stumps, max_depth=self.max_depth)
            self.estimators_generator.fit(X, neg_y)
        else:
            neg_y = None
        classification_matrix = self._binary_classification_matrix(X)
        return classification_matrix, neg_y
