"""This is the execution module, used to execute the code"""


def Exec():
    import Versions
    Versions.testVersions()
    import sys

    from MonoMultiViewClassifiers import ExecClassif
    ExecClassif.execClassif(sys.argv[1:])


if __name__=="__main__":
    Exec()