"""This is the execution module, used to execute the code"""


def exec():
    import versions
    versions.testVersions()
    import sys

    from mono_multi_view_classifiers import exec_classif
    exec_classif.execClassif(sys.argv[1:])


if __name__ == "__main__":
    exec()
