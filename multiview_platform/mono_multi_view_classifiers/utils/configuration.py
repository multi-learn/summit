import os

import yaml


def get_the_args(path_to_config_file="../config_files/config.yml"):
    """
    The function for extracting the args for a '.yml' file.

    Parameters
    ----------
    path_to_config_file : str, path to the yml file containing the configuration

    Returns
    -------
    yaml_config : dict, the dictionary conaining the configuration for the
    benchmark

    """
    with open(path_to_config_file, 'r') as stream:
        yaml_config = yaml.safe_load(stream)
    return pass_default_config(**yaml_config)


def pass_default_config(log=True,
                        name=["plausible", ],
                        label="_",
                        file_type=".hdf5",
                        views=None,
                        pathf="../data/",
                        nice=0,
                        random_state=42,
                        nb_cores=1,
                        full=True,
                        debug=False,
                        add_noise=False,
                        noise_std=0.0,
                        res_dir="../results/",
                        track_tracebacks=True,
                        split=0.49,
                        nb_folds=5,
                        nb_class=None,
                        classes=None,
                        type=["multiview", ],
                        algos_monoview=["all"],
                        algos_multiview=["svm_jumbo_fusion", ],
                        stats_iter=2,
                        metrics=["accuracy_score", "f1_score"],
                        metric_princ="accuracy_score",
                        hps_type="Random",
                        hps_iter=1,
                        hps_kwargs={'n_iter':10, "equivalent_draws":True},
                        **kwargs):
    """

    :param log:
    :param name:
    :param label:
    :param file_type:
    :param views:
    :param pathf:
    :param nice:
    :param random_state:
    :param nb_cores:
    :param full:
    :param debug:
    :param add_noise:
    :param noise_std:
    :param res_dir:
    :param track_tracebacks:
    :param split:
    :param nb_folds:
    :param nb_class:
    :param classes:
    :param type:
    :param algos_monoview:
    :param algos_multiview:
    :param stats_iter:
    :param metrics:
    :param metric_princ:
    :param hps_type:
    :param hps_iter:
    :return:
    """
    args = dict(
        (key, value) for key, value in locals().items() if key != "kwargs")
    args = dict(args, **kwargs)
    return args


def save_config(directory, arguments):
    """
    Saves the config file in the result directory.
    """
    with open(os.path.join(directory, "config_file.yml"), "w") as stream:
        yaml.dump(arguments, stream)
