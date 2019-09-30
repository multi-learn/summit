import os, sys, inspect
# from multiview_platform.mono_multi_view_classifiers.monoview_classifiers.adaboost import Adaboost


import importlib

classifier_dict = {"0": ['mono', 'Adaboost',
                             'multiview_platform.mono_multi_view_classifiers.monoview_classifiers.adaboost']}
val = classifier_dict["0"]
mymodule = importlib.import_module(val[2])

for name in dir(mymodule):
    att = getattr(mymodule, name)
    try:
        getattr(att, "__module__")
        if att.__module__.startswith(mymodule.__name__):
            if inspect.isclass(att):
                print(att)
                print(name)
    except Exception:
        pass


parameter = {"0":[]}
instring = "multiview_platform/mono_multi_view_classifiers/monoview_classifiers/"

if instring in mymodule.__file__:
    monInstance = getattr(mymodule, 'Adaboost')
    sig = inspect.signature(monInstance.__init__)
    for arg_idx, name in enumerate(sig.parameters):
        param= sig.parameters[name]
        if not name.startswith('self'):
            parameter["0"].append(name)

            if param.default is not inspect.Parameter.empty:
                value_default = param.default
            else:
                value_default = 'None'
    print()


class ConfigurationMaker():
    """
    Find the name of the classifier from the dict classier to report



    """
    _path_classifier_mono = 'multiview_platform/mono_multi_view_classifier/monoview_classifiers'
    _path_classifier_multi = 'multiview_platform/mono_multi_view_classifier/multiview_classifier'

    def __init__(self, classifier_dict=None):
        if classifier_dict is None:
            classifier_dict = {"0": ['mono', 'Adaboost',
                            'multiview_platform.mono_multi_view_classifiers.monoview_classifiers.adaboost']}
        names = []
        for key, val in  classifier_dict.items():
            mymodule = importlib.import_module(val[2])
            names.append(self._get_module_name(mymodule))


    def _get_module_name(self, mymodule):
        for name in dir(mymodule):
            att = getattr(mymodule, name)
            try:
                getattr(att, "__module__")
                if att.__module__.startswith(mymodule.__name__):
                    if inspect.isclass(att):
                        if att == val[1]:
                            return name
            except Exception:
                return None
        return None

#             mymodule = importlib.import_module(val[2])
#             module_file =  mymodule.__file__
#             getattr(self._path_classifier_mono, module_file[:-3])
#
#             #__import__(val[1], locals(), globals(), [], 1)
#             sig = inspect.signature(val[1]+"."+val[0])
#             print(sig)
#             for arg_idx, name in enumerate(sig.parameters):
#                 print(arg_idx)
#                 print(name)
#
#
# def make(dir='.', output=None):
#     """
#     Generate file config from classifier files
#     :param  dir: (default'.'
#     :dir type: str or list of str
#     :return:
#     """
#
#     currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#     parentdir = os.path.dirname(currentdir)
#     sys.path.insert(0, parentdir)
#
#
#     # calling_module = inspect.getmodule(stack_frame[0])
#
#
#
# path = os.getcwd() + '/multiview_platform/'
# files = []
# # r=root, d=directories, f = files
# for r, d, f in os.walk(path):
#     print('dir', d)
#     print('root', r)
#     for file in f:
#         if '.py' in file and not file.startswith('__init__'):
#             print("file", file)
#             files.append(os.path.join(r, file))
#
# for f in files:
#     print(f)
#
# for module in os.listdir(os.path.dirname(os.path.realpath(__file__))):
#     if module == '__init__.py' or module[-3:] != '.py':
#         continue
#     print(module)
#     __import__(module[:-3], locals(), globals(), [], 1)
#
# import glob
#
# path = 'c:\\projects\\hc2\\'
#
# files = [f for f in glob.glob(path + "**/*.txt", recursive=True)]
#
# for f in files:
#     print(f)
#
# import inspect
#
#
# # Import this to other module and call it
# def print_caller_info():
#     # Get the full stack
#     stack = inspect.stack()
#
#     # Get one level up from current
#     previous_stack_frame = stack[1]
#     print(previous_stack_frame.filename)  # Filename where caller lives
#
#     # Get the module object of the caller
#     calling_module = inspect.getmodule(stack_frame[0])
#     print(calling_module)
#     print(calling_module.__file__)
#
#
# if __name__ == '__main__':
#     print_caller_info()