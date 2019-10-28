from sklearn.metrics import pairwise
import numpy as np

from ...multiview.multiview_utils import BaseMultiviewClassifier
from ...utils.hyper_parameter_search import CustomUniform, CustomRandint

class KernelClassifier(BaseMultiviewClassifier):

    def __init__(self, random_state=None,
                 kernel_types=None, kernel_configs=None):
        super().__init__(random_state)
        self.kernel_configs=kernel_configs
        self.kernel_types=kernel_types

    def _compute_kernels(self, X, example_indices, view_indices, ):
        new_X = {}
        for index, (kernel_function, kernel_config, view_index) in enumerate(
                zip(self.kernel_functions, self.kernel_configs, view_indices)):
            new_X[index] = kernel_function(X.get_v(view_index,
                                                   example_indices),
                                           **kernel_config)
        return new_X

    def init_kernels(self, nb_view=2, ):
        if isinstance(self.kernel_types, KernelDistribution):
            self.kernel_functions = self.kernel_types.draw(nb_view)
        elif isinstance(self.kernel_types, str):
            self.kernel_functions = [getattr(pairwise, self.kernel_types)
                                     for _ in range(nb_view)]
        elif isinstance(self.kernel_types, list):
            self.kernel_functions = [getattr(pairwise, kernel_type)
                                     for kernel_type in self.kernel_types]

        if isinstance(self.kernel_configs, KernelConfigDistribution):
            self.kernel_configs = self.kernel_configs.draw(nb_view)
            self.kernel_configs = [kernel_config[kernel_function.__name__]
                                   for kernel_config, kernel_function
                                   in zip(self.kernel_configs,
                                          self.kernel_functions)]

        elif isinstance(self.kernel_configs, dict):
            self.kernel_configs = [self.kernel_configs for _ in range(nb_view)]
        else:
            pass


class KernelConfigGenerator:

    def __init__(self):
        pass

    def rvs(self, random_state=None):
        return KernelConfigDistribution(seed=random_state.randint(1))


class KernelConfigDistribution:

    def __init__(self, seed=42):
        self.random_state=np.random.RandomState(seed)
        self.possible_config = {
            "polynomial_kernel":{"degree": CustomRandint(low=1, high=7),
                                 "gamma": CustomUniform(),
                                 "coef0": CustomUniform()

            },
            "chi2_kernel": {"gamma": CustomUniform()},
            "rbf_kernel": {"gamma": CustomUniform()},
        }

    def draw(self, nb_view):
        drawn_params = [{} for _ in range(nb_view)]
        for view_index in range(nb_view):
            for kernel_name, params_dict in self.possible_config.items():
                drawn_params[view_index][kernel_name] = {}
                for param_name, distrib in params_dict.items():
                    drawn_params[view_index][kernel_name][param_name] = distrib.rvs(self.random_state)
        return drawn_params


class KernelGenerator:

    def __init__(self):
        pass

    def rvs(self, random_state=None):
        return KernelDistribution(seed=random_state.randint(1))


class KernelDistribution:

    def __init__(self, seed=42):
        self.random_state=np.random.RandomState(seed)
        self.available_kernels = [pairwise.polynomial_kernel,
                                  pairwise.chi2_kernel,
                                  pairwise.rbf_kernel,]

    def draw(self, nb_view):
        return self.random_state.choice(self.available_kernels, nb_view)
