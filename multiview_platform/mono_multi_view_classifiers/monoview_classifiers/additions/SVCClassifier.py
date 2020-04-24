from sklearn.svm import SVC


class SVCClassifier(SVC):

    def __init__(self, random_state=None, kernel='rbf', C=1.0, degree=3,
                 **kwargs):
        super(SVCClassifier, self).__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            probability=True,
            max_iter=1000,
            random_state=random_state
        )
        self.classed_params = []
        self.weird_strings = {}
