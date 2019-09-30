import numpy as np


class Parameter_pdata(object):
    class __Parameter_pdata:
        nbr_i = 0
        # option de renormalisation des donnees
        #  la séparation se faisant à une permutation pret et à un facteur de
        # renormalisation pret, on peut choisir de normaliser les données au debut
        # de l'algo et/ou à chaque iteration de l'algo et/ou à la fin de l'algo
        # on normalise A ou S
        _data_norm = {'FlagInit': True, 'FlagIter': False, 'FlagEnd': False}
        # % on normalise suivant les colonnes (1) 'dim' (norme des colonnes à 1) ou les
        # 'dim'% lignes (2) (norme des lignes à 1)
        _Norm = {'p': 1, 'dim': 1, 'x': 'A'}
        _list_mode = ['real', 'simul']
        _list_x = ['A', 'S']

        def __init__(self):
            self._Norm['p'] = 1
            self._Norm['dim'] = 1
            self._Norm['x'] = self._list_x[0]
            self.mode = self._list_mode[1]
            self.sigma = 20000
            self.dim = 1
            if self.nbr_i > 0:
                raise ValueError("Instance of class Parameter_pdata can be only one")
            self.nbr_i += 1

        def __str__(self):
            return repr(self)

    instance = None

    #     def __init__(self, arg):
    #         if not Parameter_pdata.instance:
    #             Parameter_pdata.instance = Parameter_pdata.__Parameter_pdata(arg)
    #         else:
    #             Parameter_pdata.instance.val = arg

    def __new__(cls):  # _new_ est toujours une méthode de classe
        if not Parameter_pdata.instance:
            Parameter_pdata.instance = Parameter_pdata.__Parameter_pdata()
        return Parameter_pdata.instance

    def __getattr__(self, attr):
        return getattr(self.instance, attr)

    #     def __setattr__(self, attr, val):
    #         return setattr(self.instance, attr, val)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class Parameter_palgo(object):
    class __Parameter_palgo:

        nbr_i = 0
        _list_algo = ['BCVMFB', 'PALS', 'STALS', 'LSfro', 'LSkl']
        _stop = {'DifA': False, 'DifS': False,
                 'ObjFct': True, 'threshold': np.finfo(float).eps}
        _pfwt = {'w': 'db6', 'family_pfwt': 'db',
                 'level': 10, 'K': 4,
                 'Ls': 3000, 'L1': 3000, 'L2': 3000}
        # _wavelette_type = ['db', 'db6']
        # 'LS' pour Lee et Seung
        # 'Lips' pour la constante de Lipschitz
        # 'PALM' pas de preconditionnement
        _list_precond = ['LS', 'Lips', 'PALM']

        def __init__(self):
            self.flagWave = False
            self.val = None
            algo_value = self._list_algo[1]
            self._algo = algo_value
            self.gamma = 0.99
            self.inf = np.inf
            self.eps = np.finfo(float).eps
            self.niter = 1000
            self.eta_inf = 'eps'
            self.eta_sup = 'inf'
            self.alpha_A = 0.0
            self.p_A = 1
            self.p_S = 1
            self.alpha_S = 0.0
            # self.level = 10
            self.alpha_S_eval = False
            self.stopThreshold = 10e-5,
            self.precond = 'LS'  # 'LS' pour Lee et Seung
            self.F = None
            self.Fstar = None
            self.verbose = False

            if self.nbr_i > 0:
                raise ValueError("Instance of class Parameter_pdata can be only one")
            self.nbr_i += 1

        def __str__(self):
            return repr(self) + repr(self.val)

        @property
        def algo(self):
            return self._algo

        @algo.setter
        def algo(self, algo_value):
            if algo_value not in self._list_algo:
                raise NameError("parameter algo must be in %s" % self._list_algo)
            else:
                self._algo = algo_value

    instance = None

    #     def __init__(self, arg):
    #         if not Parameter_pdata.instance:
    #             Parameter_pdata.instance = Parameter_pdata.__Parameter_pdata(arg)
    #         else:
    #             Parameter_pdata.instance.val = arg

    def __new__(cls):  # _new_ est toujours une méthode de classe
        if not Parameter_palgo.instance:
            Parameter_palgo.instance = Parameter_palgo.__Parameter_palgo()
        return Parameter_palgo.instance

    def __getattr__(self, attr):
        return getattr(self.instance, attr)

    #     def __setattr__(self, attr, val):
    #         return setattr(self.instance, attr, val)

    def __setattr__(self, name):
        return setattr(self.instance, name)


if __name__ == '__main__':
    a = Parameter_pdata()
    a = Parameter_pdata()
    b = Parameter_pdata()
    b.val = 6
    b.x = 8
    a.x = 10
    param = Parameter_palgo()
    algo = param._list_algo[3]
    param.algo = algo
