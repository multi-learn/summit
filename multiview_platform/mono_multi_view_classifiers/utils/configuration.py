import yaml
import os


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
    return yaml_config


def save_config(directory, arguments):
    """
    Saves the config file in the result directory.
    """
    with open(os.path.join(directory, "config_file.yml"), "w") as stream:
        yaml.dump(arguments, stream)