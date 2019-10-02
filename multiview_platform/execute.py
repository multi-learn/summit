"""This is the execution module, used to execute the code"""

<<<<<<< HEAD

def execute():
    import multiview_platform.versions as vs
    vs.test_versions()
=======
def exec():
    import multiview_platform.versions as versions
    versions.test_versions()
>>>>>>> 7b3e918b4fb2938657cae3093d95b1bd6fc461d4
    import sys

    from multiview_platform.mono_multi_view_classifiers import exec_classif
    exec_classif.exec_classif(sys.argv[1:])


if __name__ == "__main__":
    execute()
