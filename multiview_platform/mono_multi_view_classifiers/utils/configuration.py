import configparser
import builtins
from distutils.util import strtobool as tobool


def get_the_args(path_to_config_file="../config_files/config.ini"):
    """This is the main function for extracting the args for a '.ini' file"""
    config_parser = configparser.ConfigParser(comment_prefixes=('#'))
    config_parser.read(path_to_config_file)
    config_dict = {}
    for section in config_parser:
        config_dict[section] = {}
        for key in config_parser[section]:
            value = format_raw_arg(config_parser[section][key])
            config_dict[section][key] = value
    return config_dict


def format_raw_arg(raw_arg):
    """This function is used to convert the raw arg in a types value.
    For example, 'list_int ; 10 20' will be formatted in [10,20]"""
    function_name, raw_value = raw_arg.split(" ; ")
    if function_name.startswith("list"):
        function_name = function_name.split("_")[1]
        raw_values = raw_value.split(" ")
        value = [getattr(builtins, function_name)(raw_value)
                 if function_name != "bool" else bool(tobool(raw_value))
                 for raw_value in raw_values]
    else:
        if raw_value == "None":
            value = None
        else:
            if function_name=="bool":
                value = bool(tobool(raw_value))
            else:
                value = getattr(builtins, function_name)(raw_value)
    return value
