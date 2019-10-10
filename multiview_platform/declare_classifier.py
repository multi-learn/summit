import os
import importlib
import inspect


class ClassierMakerMultiViewPlatform():
    _benchmark = {"monoview":
                      {"path_classifier": 'multiview_platform/mono_multi_view_classifier/monoview_classifiers'},
                  "multiview":
                      {"path_classifier_multi": 'multiview_platform/mono_multi_view_classifier/multiview_classifier'}}


    def __init__(self, classifier_names, classifier_modules=None, classifier_files=None, mod='monoview'):
        if classifier_files is None and classifier_names.size != classifier_modules.size:
            raise ValueError("attr classifier_names and  classifier_modules should have same size")
        if classifier_modules is None and classifier_names.size != classifier_files.size:
            raise ValueError("attr classifier_names and  classifier_files should have same size")

        if classifier_files is None:
            for classifier, module in zip(classifier_names, classifier_modules):
                my_instance, my_module = self._check_classifier_install

                self._create_class(my_instance, my_module)


    def _check_classifier_install(self, classifier, module):
        try:
            my_module = importlib.import_module(module)
        except Exception:
            raise("the module %d can't be imported" % module)
        try:
            my_instance = getattr(my_module, classifier)
        except AttributeError:
            raise AttributeError("The class %d is not in %d" % classifier  %module)
        return my_instance, my_module

    def _create_class(self, classifier, module):
        if mod.startswith('monoview'):
            directory =  self._benchmark[mod]["path_classifier"]


    def _get_module_name(self, mymodule):
        for name in dir(mymodule):
            att = getattr(mymodule, name)
            try:
                getattr(att, "__module__")
                if att.__module__.startswith(mymodule.__name__):
                    if inspect.isclass(att):
                        if att == name:
                            return name
            except Exception:
                return None
        return None