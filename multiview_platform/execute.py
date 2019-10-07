"""This is the execution module, used to execute the code"""


def execute():
    from multiview_platform import versions as vs
    vs.test_versions()
    import sys

    from multiview_platform.mono_multi_view_classifiers import exec_classif
    exec_classif.exec_classif(sys.argv[1:])


if __name__ == "__main__":
    execute()
