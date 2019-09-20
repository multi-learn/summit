from .cg_desc import CGDesc


class CGDesc10(CGDesc):

    def __init__(self, random_state=None, n_max_iterations=500, n_stumps=1,
                 **kwargs):
        super(CGDesc10, self).__init__(n_max_iterations=100,
                                       random_state=random_state,
                                       n_stumps=10, )


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_stumps": args.CGD_stumps,
                  "n_max_iterations": args.CGD_n_iter}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
