"""This is the execution module, used to execute the code"""

import os

def execute(config_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "config_files", "config_example_1.yml")):
    from multiview_platform import versions as vs
    vs.test_versions()
    import sys

    from multiview_platform.mono_multi_view_classifiers import exec_classif
    if sys.argv[1:]:
        exec_classif.exec_classif(sys.argv[1:])
    else:
        if config_path == "example 0":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "config_files", "config_example_0.yml")
        elif config_path == "example 1":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "config_files", "config_example_1.yml")
        elif config_path == "example 2.1.1":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "config_files", "config_example_2_1_1.yml")
        elif config_path == "example 2.1.2":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "config_files", "config_example_2_1_2.yml")
        elif config_path == "example 2.2":
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "examples", "config_files", "config_example_2_2.yml")
        exec_classif.exec_classif(["--config_path", config_path])


if __name__ == "__main__":
    execute()
