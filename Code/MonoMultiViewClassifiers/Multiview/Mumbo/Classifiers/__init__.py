# from os import listdir
# from os.path import isfile, join
# mypath="."
# modules = [f[:-3] for f in listdir(mypath) if isfile(join(mypath, f)) and f[-3:] == ".py" and f!="__init__.py" ]
# __all__ = modules

import os
for module in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__(module[:-3], locals(), globals(), [], 1)
del module
del os
