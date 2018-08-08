import os

for module in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if module == '__init__.py' or module[-3:] == '.py' or module[-4:] == '.pyc' or module == '__pycache__' :
        continue
    __import__(module, locals(), globals(), [], 1)
del module
del os
