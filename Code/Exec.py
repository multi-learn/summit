if __name__=="__main__":
    import Versions
    Versions.testVersions()
    import sys

    from MonoMultiViewClassifiers import ExecClassif
    ExecClassif.execClassif(sys.argv[1:])


