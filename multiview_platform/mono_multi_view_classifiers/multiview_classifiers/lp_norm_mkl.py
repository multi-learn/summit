
from sklearn.metrics import pairwise

from ..multiview.multiview_utils import BaseMultiviewClassifier, get_examples_views_indices
from .additions.kernel_learning import KernelClassifier, KernelConfigGenerator, KernelGenerator
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint

classifier_class_name = "LPNormMKL"

### The following code is a welcome contribution by Riikka Huusari
# (riikka.huusari@lis-lab.fr) that we adapted te create the classifier


import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y
from .additions.data_sample import Metriclearn_array


class MKL(BaseEstimator, ClassifierMixin):
    def __init__(self, lmbda, m_param=1.0, use_approx=True, max_rounds=50,
                 max_diff=0.0001, p=2):
        print(lmbda)
        # calculate nyström approximation (if used)
        self.lmbda = lmbda
        self.use_approx = use_approx
        self.m_param = m_param

        # Non-optimizable Hyper-params
        self.max_rounds = max_rounds
        self.max_diff = max_diff
        self.p = p

    def fit(self, X, y= None, views_ind=None):
        if isinstance(X, Metriclearn_array):
            self.X_ = X
        elif isinstance(X, dict):
            self.X_ = Metriclearn_array(X)
        elif isinstance(X, np.ndarray) :
            self.X_ = Metriclearn_array(X, views_ind)
        self.classes_ = unique_labels(y)
        check_X_y(self.X_, y)
        self.y_ = y
        n = self.X_.shape[0]
        self._calc_nystrom(self.X_, n)
        C, weights = self.learn_lpMKL()
        self.C = C
        self.weights = weights

    def learn_lpMKL(self):

        views = self.X_.n_views
        X = self.X_
        # p = 2
        n = self.X_.shape[0]
        weights = np.ones(views) / (views)

        prevalpha = False
        max_diff = 1

        kernels = np.zeros((views, n, n))
        for v in range(0, views):
            kernels[v, :, :] = np.dot(self.U_dict[v], np.transpose(self.U_dict[v]))

        rounds = 0
        stuck = False
        while max_diff > self.max_diff and rounds < self.max_rounds and not stuck:

            # gammas are fixed upon arrival to the loop
            # -> solve for alpha!

            if self.m_param < 1 and self.use_approx:
                combined_kernel = np.zeros((n, n))
                for v in range(0, views):
                    combined_kernel = combined_kernel + weights[v] * kernels[v]
            else:
                combined_kernel = np.zeros((n, n))
                for v in range(0, views):
                    combined_kernel = combined_kernel + weights[v]*X.get_view(v)
            # combined kernel includes the weights

            # alpha = (K-lambda*I)^-1 y
            C = np.linalg.solve((combined_kernel + self.lmbda * np.eye(n)), self.y_)

            # alpha fixed -> calculate gammas
            weights_old = weights.copy()

            # first the ||f_t||^2 todo wtf is the formula used here????
            ft2 = np.zeros(views)
            for v in range(0, views):
                if self.m_param < 1 and self.use_approx:
                        # ft2[v,vv] = weights_old[v,vv] * np.dot(np.transpose(C), np.dot(np.dot(np.dot(data.U_dict[v],
                        #                                                             np.transpose(data.U_dict[v])),
                        #                                                             np.dot(data.U_dict[vv],
                        #                                                             np.transpose(data.U_dict[vv]))), C))
                    ft2[v] = np.linalg.norm(weights_old[v] * np.dot(kernels[v], C))**2
                else:
                    ft2[v] = np.linalg.norm(weights_old[v] * np.dot(X.get_view(v), C))**2
                    # ft2[v] = weights_old[v] * np.dot(np.transpose(C), np.dot(data.kernel_dict[v], C))

            # calculate the sum for downstairs

            # print(weights_old)
            # print(ft2)
            # print(ft2 ** (p / (p + 1.0)))

            downstairs = np.sum(ft2 ** (self.p / (self.p + 1.0))) ** (1.0 / self.p)
            # and then the gammas
            weights = (ft2 ** (1 / (self.p + 1))) / downstairs

            # convergence
            if prevalpha == False:  # first time in loop we don't have a previous alpha value
                prevalpha = True
                diff_alpha = 1
            else:
                diff_alpha = np.linalg.norm(C_old - C) / np.linalg.norm(C_old)
                max_diff_gamma_prev = max_diff_gamma

            max_diff_gamma = np.max(np.max(np.abs(weights - weights_old)))

            # try to see if convergence is as good as it gets: if it is stuck
            if max_diff_gamma < 1e-3 and max_diff_gamma_prev < max_diff_gamma:
                # if the gamma difference starts to grow we are most definitely stuck!
                # (this condition determined empirically by running algo and observing the convergence)
                stuck = True
            if rounds > 1 and max_diff_gamma - max_diff_gamma_prev > 1e-2:
                # If suddenly the difference starts to grow much
                stuck = True

            max_diff = np.max([max_diff_gamma, diff_alpha])
            # print([max_diff_gamma, diff_alpha])  # print if convergence is interesting
            C_old = C.copy()
            rounds = rounds + 1

        # print("\nlearned the weights:")
        # np.set_printoptions(precision=3, suppress=True)
        # print(weights)
        # print("")

        # print if resulting convergence is of interest
        # print("convergence of ", max_diff, " at step ", rounds, "/500")

        if stuck:
            return C_old, weights_old
        else:
            return C, weights


    def predict(self, X, views_ind=None):
        if isinstance(X, Metriclearn_array):
            # self.X_ = X
            pass
        elif isinstance(X, dict):
            X = Metriclearn_array(X)
        elif isinstance(X, np.ndarray):
            X = Metriclearn_array(X, views_ind)
        C = self.C
        weights  = self.weights
        return self.lpMKL_predict(X , C, weights)


    def lpMKL_predict(self, X, C, weights, views_ind=None):
        if isinstance(X, Metriclearn_array):
            # self.X_ = X
            pass
        elif isinstance(X, dict):
            X = Metriclearn_array(X)
        elif isinstance(X, np.ndarray):
            X = Metriclearn_array(X, views_ind)
        views = X.n_views
        tt = X.shape[0]
        m = self.X_.shape[0] # self.m_param * n

        #  NO TEST KERNEL APPROXIMATION
        # kernel = weights[0] * self.data.test_kernel_dict[0]
        # for v in range(1, views):
        #     kernel = kernel + weights[v] * self.data.test_kernel_dict[v]

        # TEST KERNEL APPROXIMATION
        kernel = np.zeros((tt, self.X_.shape[0]))
        for v in range(0, views):
            if self.m_param < 1:
                kernel = kernel + weights[v] * np.dot(np.dot(X.get_view(v)[:, 0:m], self.W_sqrootinv_dict[v]),
                                                  np.transpose(self.U_dict[v]))
            else:
                kernel = kernel + weights[v] * X.get_view(v)

        return np.dot(kernel, C)

    def _calc_nystrom(self, kernels, n_approx):
        # calculates the nyström approximation for all the kernels in the given dictionary
        self.W_sqrootinv_dict = {}
        self.U_dict = {}
        for v in range(kernels.n_views):
            kernel = kernels.get_view(v)
            E = kernel[:, 0:n_approx]
            W = E[0:n_approx, :]
            Ue, Va, _ = np.linalg.svd(W)
            vak = Va[0:n_approx]
            inVa = np.diag(vak ** (-0.5))
            U_v = np.dot(E, np.dot(Ue[:, 0:n_approx], inVa))
            self.U_dict[v] = U_v
            self.W_sqrootinv_dict[v] = np.dot(Ue[:, 0:n_approx], inVa)


class LPNormMKL(KernelClassifier, MKL):
    def __init__(self, random_state=None, lmbda=0.1, m_param=1, max_rounds=50,
                 max_diff=0.0001, use_approx=True, kernel_types="rbf_kernel",
                 kernel_configs=None, p=2, prev_alpha=False):
        super().__init__(random_state, kernel_configs=kernel_configs,
                         kernel_types=kernel_types)
        super(BaseMultiviewClassifier, self).__init__(lmbda, m_param,
                                                      use_approx, max_rounds,
                                                      max_diff, p)
        self.param_names = ["lmbda", "kernel_types", "kernel_configs"]
        self.distribs = [CustomUniform(), KernelGenerator(),
                         KernelConfigGenerator()]

        self.prev_alpha = prev_alpha

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X, train_indices,
                                                                 view_indices)
        self.init_kernels(nb_view= len(view_indices), )
        new_X = self._compute_kernels(X,
                                      train_indices, view_indices)
        return super(LPNormMKL, self).fit(new_X, y[train_indices])

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        new_X = self._compute_kernels(X, example_indices, view_indices)
        return super(LPNormMKL, self).predict(new_X)



