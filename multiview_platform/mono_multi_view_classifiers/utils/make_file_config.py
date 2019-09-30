import os, sys, inspect
# from multiview_platform.mono_multi_view_classifiers.monoview_classifiers.adaboost import Adaboost


import importlib

#
# if instring in mymodule.__file__:
#
#     sig = inspect.signature(monInstance.__init__)
#     for arg_idx, name in enumerate(sig.parameters):
#         param= sig.parameters[name]
#         if not name.startswith('self'):
#             parameter["0"].append(name)
#
#             if param.default is not inspect.Parameter.empty:
#                 value_default = param.default
#             else:
#                 value_default = 'None'
#     print()


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
            monInstance = getattr(mymodule, val[1])


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


if __name__ == '__main__':
     ConfigurationMaker()
