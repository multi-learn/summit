import os, sys, inspect
from multiview_platform.mono_multi_view_classifiers.monoview_classifiers.adaboost import Adaboost



class ConfigurationMaker():
    """

    """

    def __init__(self):
    classifier_dict = {"0": [Adaboost,
                             multiview_platform.mono_multi_view_classifiers.monoview_classifiers.adaboost]}
        for key, val in  classifier_dict.items():

           __import__(val[1], locals(), globals(), [], 1)
           sig = inspect.signature(val[1]+"."+val[0]+".__init__")
           print(sig)
           for arg_idx, name in enumerate(sig.parameters):
               print(arg_idx)
               print(name)


def make(dir='.', output=None):
    """
    Generate file config from classifier files
    :param  dir: (default'.'
    :dir type: str or list of str
    :return:
    """

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)

    import mymodule
     calling_module = inspect.getmodule(stack_frame[0])




import os

path = os.getcwd() + '/multiview_platform/'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    print('dir', d)
    print('root', r)
    for file in f:
        if '.py' in file and not file.startswith('__init__'):
            print("file", file)
            files.append(os.path.join(r, file))

for f in files:
    print(f)

for module in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    __import__(module[:-3], locals(), globals(), [], 1)

import glob

path = 'c:\\projects\\hc2\\'

files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]

for f in files:
    print(f)

import inspect


# Import this to other module and call it
def print_caller_info():
    # Get the full stack
    stack = inspect.stack()

    # Get one level up from current
    previous_stack_frame = stack[1]
    print(previous_stack_frame.filename)  # Filename where caller lives

    # Get the module object of the caller
    calling_module = inspect.getmodule(stack_frame[0])
    print(calling_module)
    print(calling_module.__file__)


if __name__ == '__main__':
    print_caller_info()