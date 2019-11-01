import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation  import check_array
from sklearn.utils.validation  import check_is_fitted
from .additions.data_sample import DataSample, Metriclearn_array

"""
    Copyright (C) 2018  Riikka Huusari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


This file contains algorithms for Multi-View Metric Learning (MVML) as introduced in

Riikka Huusari, Hachem Kadri and Cécile Capponi:
Multi-View Metric Learning in Vector-Valued Kernel Spaces
in International Conference on Artificial Intelligence and Statistics (AISTATS) 2018

Usage (see also demo.py for a more detailed example):
    create a MVML object via:
        mvml = MVML(kernel_dict, label_vector, regression_parameter_list, nystrom_param)
    learn the model:
        A, g, w = mvml.learn_mvml()
    predict with the model:
        predictions = predict_mvml(test_kernel_dict, g, w)

(parameter names as in the paper)

Code is tested with Python 3.5.2 and numpy 1.12.1
"""


class MVML(BaseEstimator, ClassifierMixin):
    r"""
    The MVML Classifier

    Parameters
    ----------
    regression_params: array/list of regression parameters, first for basic regularization, second for
        regularization of A (not necessary if A is not learned)

    nystrom_param: value between 0 and 1 indicating level of nyström approximation; 1 = no approximation

    learn_A :  integer (default 1) choose if A is learned or not: 1 - yes (default);
               2 - yes, sparse; 3 - no (MVML_Cov); 4 - no (MVML_I)

    learn_w : integer (default 0) where learn w is needed

    n_loops : (default 0) number of itterions


    Attributes
    ----------
    reg_params : array/list of regression parameters

    learn_A :  1 where Learn matrix A is needded

    learn_w : integer where learn w is needed

    n_loops : number of itterions

    n_approx : number of samples in approximation, equals n if no approx.

    X_ : :class:`metriclearning.datasets.data_sample.Metriclearn_array` array of input sample

    y_ : array-like, shape = (n_samples,)
         Target values (class labels).

    """

    def __init__(self, regression_params, nystrom_param, learn_A=1, learn_w=0, n_loops=6):

        # calculate nyström approximation (if used)
        self.nystrom_param = nystrom_param

        self.reg_params = regression_params
        self.learn_A = learn_A
        self.learn_w = learn_w
        self.n_loops = n_loops

    def fit(self, X, y= None, views_ind=None):
        """
        Fit the MVML classifier
        Parameters
        ----------

        X : Metriclearn_array {array-like, sparse matrix}, shape = (n_samples, n_features)
            Training multi-view input samples.


        y : array-like, shape = (n_samples,)
            Target values (class labels).
            array of length n_samples containing the classification/regression labels
            for training data

        views_ind : array-like (default=[0, n_features//2, n_features])
            Paramater specifying how to extract the data views from X:

            - If views_ind is a 1-D array of sorted integers, the entries
              indicate the limits of the slices used to extract the views,
              where view ``n`` is given by
              ``X[:, views_ind[n]:views_ind[n+1]]``.

              With this convention each view is therefore a view (in the NumPy
              sense) of X and no copy of the data is done.


        Returns
        -------

        self : object
            Returns self.
        """
        # Check that X and y have correct shape

        # Store the classes seen during fit
        if isinstance(X, Metriclearn_array):
            self.X_ = X
        elif isinstance(X, np.ndarray) :
            self.X_= Metriclearn_array(X, views_ind)
        elif isinstance(X, dict):
            self.X_= Metriclearn_array(X)
        else:
            raise TypeError("Input format is not reconized")
        check_X_y(self.X_, y)
        self.classes_ = unique_labels(y)
        self.y_ = y

        # n = X[0].shape[0]
        n = self.X_.shape[0]
        self.n_approx = int(np.floor(self.nystrom_param * n))  # number of samples in approximation, equals n if no approx.

        if self.nystrom_param < 1:
            self._calc_nystrom(self.X_)
        else:
            self.U_dict = self.X_.to_dict()

        # Return the classifier
        self.learn_mvml(learn_A=self.learn_A, learn_w=self.learn_w, n_loops=self.n_loops)
        return self

    def learn_mvml(self, learn_A=1, learn_w=0, n_loops=6):
        """

        Parameters
        ----------
        learn_A: int choose if A is learned or not (default: 1):
                 1 - yes (default);
                 2 - yes, sparse;
                 3 - no (MVML_Cov);
                 4 - no (MVML_I)
        learn_w: int choose if w is learned or not (default: 0):
                 0 - no (uniform 1/views, default setting),
                 1 - yes
        n_loops: int maximum number of iterations in MVML, (default: 6)
                 usually something like default 6 is already converged

        Returns
        -------
        tuple (A, g, w) with A (metrcic matrix - either fixed or learned),
                             g (solution to learning problem),
                             w (weights - fixed or learned)
        """
        views = len(self.U_dict)
        n = self.U_dict[0].shape[0]
        lmbda = self.reg_params[0]
        if learn_A < 3:
            eta = self.reg_params[1]

        # ========= initialize A =========

        # positive definite initialization (with multiplication with the U matrices if using approximation)
        A = np.zeros((views * self.n_approx, views * self.n_approx))
        if learn_A < 3:
            for v in range(views):
                if self.nystrom_param < 1:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.dot(np.transpose(self.U_dict[v]), self.U_dict[v])
                else:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = np.eye(n)
        # otherwise initialize like this if using MVML_Cov
        elif learn_A == 3:
            for v in range(views):
                for vv in range(views):
                    if self.nystrom_param < 1:
                        A[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                            np.dot(np.transpose(self.U_dict[v]), self.U_dict[vv])
                    else:
                        A[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                            np.eye(n)
        # or like this if using MVML_I
        elif learn_A == 4:
            for v in range(views):
                if self.nystrom_param < 1:
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.eye(self.n_approx)
                else:
                    # it might be wise to make a dedicated function for MVML_I if using no approximation
                    # - numerical errors are more probable this way using inverse
                    A[v * self.n_approx:(v + 1) * self.n_approx, v * self.n_approx:(v + 1) * self.n_approx] = \
                        np.linalg.pinv(self.U_dict[v])  # U_dict holds whole kernels if no approx

        # ========= initialize w, allocate g =========
        w = (1 / views) * np.ones((views, 1))
        g = np.zeros((views * self.n_approx, 1))

        # ========= learn =========
        loop_counter = 0
        while True:

            if loop_counter > 0:
                g_prev = np.copy(g)
                A_prev = np.copy(A)
                w_prev = np.copy(w)

            # ========= update g =========

            # first invert A
            try:
                A_inv = np.linalg.pinv(A + 1e-09 * np.eye(views * self.n_approx))
            except np.linalg.linalg.LinAlgError:
                try:
                    A_inv = np.linalg.pinv(A + 1e-06 * np.eye(views * self.n_approx))
                except ValueError:
                    return A_prev, g_prev
            except ValueError:
                return A_prev, g_prev

            # then calculate g (block-sparse multiplications in loop) using A_inv
            for v in range(views):
                for vv in range(views):
                    A_inv[v * self.n_approx:(v + 1) * self.n_approx, vv * self.n_approx:(vv + 1) * self.n_approx] = \
                        w[v] * w[vv] * np.dot(np.transpose(self.U_dict[v]), self.U_dict[vv]) + \
                        lmbda * A_inv[v * self.n_approx:(v + 1) * self.n_approx,
                                      vv * self.n_approx:(vv + 1) * self.n_approx]
                g[v * self.n_approx:(v + 1) * self.n_approx, 0] = np.dot(w[v] * np.transpose(self.U_dict[v]), self.y_)

            try:
                g = np.dot(np.linalg.pinv(A_inv), g)  # here A_inv isn't actually inverse of A (changed in above loop)
            except np.linalg.linalg.LinAlgError:
                g = np.linalg.solve(A_inv, g)

            # ========= check convergence =========

            if learn_A > 2 and learn_w != 1:  # stop at once if only g is to be learned
                break

            if loop_counter > 0:

                # convergence criteria
                g_diff = np.linalg.norm(g - g_prev) / np.linalg.norm(g_prev)
                A_diff = np.linalg.norm(A - A_prev, ord='fro') / np.linalg.norm(A_prev, ord='fro')
                if g_diff < 1e-4 and A_diff < 1e-4:
                    break

            if loop_counter >= n_loops:  # failsafe
                break

            # ========= update A =========
            if learn_A == 1:
                A = self._learn_A_func(A, g, lmbda, eta)
            elif learn_A == 2:
                A = self._learn_blocksparse_A(A, g, views, self.n_approx, lmbda, eta)

            # ========= update w =========
            if learn_w == 1:
                Z = np.zeros((n, views))
                for v in range(views):
                    Z[:, v] = np.dot(self.U_dict[v], g[v * self.n_approx:(v + 1) * self.n_approx]).ravel()
                w = np.dot(np.linalg.pinv(np.dot(np.transpose(Z), Z)), np.dot(np.transpose(Z), self.y_))

            loop_counter += 1
        self.g = g
        self.w = w
        self.A = A
        return A, g, w


    def predict(self, X, views_ind=None):
        """

        Parameters
        ----------
        X

        Returns
        -------

        """

        """


        :param X:
        :return: 
        """
        if isinstance(X, Metriclearn_array):
            self.X_ = X
        elif isinstance(X, np.ndarray) :
            self.X_= Metriclearn_array(X, views_ind)
        elif isinstance(X, dict):
            self.X_= Metriclearn_array(X)
        else:
            raise TypeError("Input format is not reconized")
        check_is_fitted(self, ['X_', 'y_'])
        check_array(self.X_)
        check_is_fitted(self, ['X_', 'y_'])
        return self.predict_mvml(self.X_, self.g, self.w)

    def predict_mvml(self, test_kernels, g, w):

        """
        :param test_kernels: dictionary of test kernels (as the dictionary of kernels in __init__)
        :param g: g, learning solution that is learned in learn_mvml
        :param w: w, weights for combining the solutions of views, learned in learn_mvml
        :return: (regression) predictions, array of size  test_samples*1
        """

        views = len(self.U_dict)
        # t = test_kernels[0].shape[0]
        t = test_kernels.shape[0]
        X = np.zeros((t, views * self.n_approx))
        for v in range(views):
            if self.nystrom_param < 1:
                X[:, v * self.n_approx:(v + 1) * self.n_approx] = w[v] * \
                                                                  np.dot(test_kernels.get_view(v)[:, 0:self.n_approx],
                                                                         self.W_sqrootinv_dict[v])
            else:
                X[:, v * self.n_approx:(v + 1) * self.n_approx] = w[v] * test_kernels[v]

        return np.dot(X, g)

    def _calc_nystrom(self, kernels):
        # calculates the nyström approximation for all the kernels in the given dictionary
        self.W_sqrootinv_dict = {}
        self.U_dict = {}
        for v in range(len(kernels.shapes_int)):
            kernel = kernels.get_view(v)
            E = kernel[:, 0:self.n_approx]
            W = E[0:self.n_approx, :]
            Ue, Va, _ = np.linalg.svd(W)
            vak = Va[0:self.n_approx]
            inVa = np.diag(vak ** (-0.5))
            U_v = np.dot(E, np.dot(Ue[:, 0:self.n_approx], inVa))
            self.U_dict[v] = U_v
            self.W_sqrootinv_dict[v] = np.dot(Ue[:, 0:self.n_approx], inVa)

    def _learn_A_func(self, A, g, lmbda, eta):

        # basic gradient descent

        stepsize = 0.5
        if stepsize*eta >= 0.5:
            stepsize = 0.9*(1/(2*eta))  # make stepsize*eta < 0.5

        loops = 0
        not_converged = True
        while not_converged:

            A_prev = np.copy(A)

            A_pinv = np.linalg.pinv(A)
            A = (1-2*stepsize*eta)*A + stepsize*lmbda*np.dot(np.dot(A_pinv, g), np.dot(np.transpose(g), A_pinv))

            if loops > 0:
                prev_diff = diff
            diff = np.linalg.norm(A - A_prev) / np.linalg.norm(A_prev)

            if loops > 0 and prev_diff > diff:
                A = A_prev
                stepsize = stepsize*0.1

            if diff < 1e-5:
                not_converged = False

            if loops > 10:
                not_converged = False

            loops += 1

        return A

    def _learn_blocksparse_A(self, A, g, views, m, lmbda, eta):

        # proximal gradient update method

        converged = False
        rounds = 0

        L = lmbda * np.linalg.norm(np.dot(g, g.T))
        # print("L ", L)

        while not converged and rounds < 100:

            # no line search - this has worked well enough experimentally
            A = self._proximal_update(A, views, m, L, g, lmbda, eta)

            # convergence
            if rounds > 0:
                A_diff = np.linalg.norm(A - A_prev) / np.linalg.norm(A_prev)

                if A_diff < 1e-3:
                    converged = True

            A_prev = np.copy(A)

            rounds += 1

        return A

    def _proximal_update(self, A_prev, views, m, L, D, lmbda, gamma):

        # proximal update

        # the inverse is not always good to compute - in that case just return the previous one and end the search
        try:
            A_prev_inv = np.linalg.pinv(A_prev)
        except np.linalg.linalg.LinAlgError:
            try:
                A_prev_inv = np.linalg.pinv(A_prev + 1e-6 * np.eye(views * m))
            except np.linalg.linalg.LinAlgError:
                return A_prev
            except ValueError:
                return A_prev
        except ValueError:
            return A_prev

        if np.any(np.isnan(A_prev_inv)):
            # just in case the inverse didn't return a proper solution (happened once or twice)
            return A_prev

        A_tmp = A_prev + (lmbda / L) * np.dot(np.dot(A_prev_inv.T, D), np.dot(np.transpose(D), A_prev_inv.T))

        # if there is one small negative eigenvalue this gets rid of it
        try:
            val, vec = np.linalg.eigh(A_tmp)
        except np.linalg.linalg.LinAlgError:
            return A_prev
        except ValueError:
            return A_prev
        val[val < 0] = 0

        A_tmp = np.dot(vec, np.dot(np.diag(val), np.transpose(vec)))
        A_new = np.zeros((views*m, views*m))

        # proximal update, group by group (symmetric!)
        for v in range(views):
            for vv in range(v + 1):
                if v != vv:
                    if np.linalg.norm(A_tmp[v * m:(v + 1) * m, vv * m:(vv + 1) * m]) != 0:
                        multiplier = 1 - gamma / (2 * np.linalg.norm(A_tmp[v * m:(v + 1) * m, vv * m:(vv + 1) * m]))
                        if multiplier > 0:
                            A_new[v * m:(v + 1) * m, vv * m:(vv + 1) * m] = multiplier * A_tmp[v * m:(v + 1) * m,
                                                                                               vv * m:(vv + 1) * m]
                            A_new[vv * m:(vv + 1) * m, v * m:(v + 1) * m] = multiplier * A_tmp[vv * m:(vv + 1) * m,
                                                                                               v * m:(v + 1) * m]
                else:
                    if (np.linalg.norm(A_tmp[v * m:(v + 1) * m, v * m:(v + 1) * m])) != 0:
                        multiplier = 1 - gamma / (np.linalg.norm(A_tmp[v * m:(v + 1) * m, v * m:(v + 1) * m]))
                        if multiplier > 0:
                            A_new[v * m:(v + 1) * m, v * m:(v + 1) * m] = multiplier * A_tmp[v * m:(v + 1) * m,
                                                                                             v * m:(v + 1) * m]

        return A_new


from ..multiview.multiview_utils import BaseMultiviewClassifier, get_examples_views_indices
from .additions.kernel_learning import KernelClassifier, KernelConfigGenerator, KernelGenerator
from ..utils.hyper_parameter_search import CustomUniform, CustomRandint

classifier_class_name = "MVMLClassifier"

class MVMLClassifier(KernelClassifier, MVML):

    def __init__(self, random_state=None, reg_params=None,
                 nystrom_param=1, learn_A=1, learn_w=0, n_loops=6, kernel_types="rbf_kernel",
                 kernel_configs=None):
        super().__init__(random_state, kernel_types=kernel_types,
                         kernel_configs=kernel_configs)
        super(BaseMultiviewClassifier, self).__init__(reg_params,
                                                      nystrom_param,
                                                      learn_A=learn_A,
                                                      learn_w=learn_w,
                                                      n_loops=n_loops)
        self.param_names = ["nystrom_param", "kernel_types", "kernel_configs",
                            "learn_A", "learn_w", "n_loops", "reg_params"]
        self.distribs = [CustomUniform(), KernelGenerator(),
                         KernelConfigGenerator(), CustomRandint(low=1, high=5),
                         [0,1], CustomRandint(low=1, high=100), [[0.1,0.9]]]

    def fit(self, X, y, train_indices=None, view_indices=None):
        new_X, new_y = self._init_fit(X, y, train_indices, view_indices)
        return super(MVMLClassifier, self).fit(new_X, new_y)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                   example_indices,
                                                                   view_indices)
        new_X = self._compute_kernels(X, example_indices, view_indices)
        print(self.extract_labels(super(MVMLClassifier, self).predict(new_X)))
        return self.extract_labels(super(MVMLClassifier, self).predict(new_X))

