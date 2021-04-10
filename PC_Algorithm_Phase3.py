#######################################################################################################################
from copy import deepcopy
#######################################################################################################################


def Meek(cg):
    """ Run Meek rules
    :param cg: a CausalGraph object
    :return:
    cg_new: a CausalGraph object
    """
    cg_new = deepcopy(cg)

    UT = cg_new.findUnshieldedTriples()
    Tri = cg_new.findTriangles()
    Kite = cg_new.findKites()

    Loop = True

    while Loop:
        Loop = False
        for (i, j, k) in UT:
            if cg_new.isFullyDirected(i, j) and cg_new.isUndirected(j, k):
                cg_new.adjmat[j, k] = 1
                Loop = True

        for (i, j, k) in Tri:
            if cg_new.isFullyDirected(i, j) and cg_new.isFullyDirected(j, k) and cg_new.isUndirected(i, k):
                cg_new.adjmat[i, k] = 1
                Loop = True

        for (i, j, k, l) in Kite:
            if cg_new.isUndirected(i, j) and cg_new.isUndirected(i, k) and cg_new.isFullyDirected(j, l)\
                    and cg_new.isFullyDirected(k, l) and cg_new.isUndirected(i, l):
                cg_new.adjmat[i, l] = 1
                Loop = True

    return cg_new

#######################################################################################################################


def definite_Meek(cg):
    """ Run Meek rules over the definite unshielded triples
    :param cg: a CausalGraph object
    :return:
    cg_new: a CausalGraph object
    """
    cg_new = deepcopy(cg)

    Tri = cg_new.findTriangles()
    Kite = cg_new.findKites()

    Loop = True

    while Loop:
        Loop = False
        for (i, j, k) in cg_new.definite_non_UC:
            if cg_new.isFullyDirected(i, j) and cg_new.isUndirected(j, k):
                cg_new.adjmat[j, k] = 1
                Loop = True
            elif cg_new.isFullyDirected(k, j) and cg_new.isUndirected(j, i):
                cg_new.adjmat[j, i] = 1
                Loop = True

        for (i, j, k) in Tri:
            if cg_new.isFullyDirected(i, j) and cg_new.isFullyDirected(j, k) and cg_new.isUndirected(i, k):
                cg_new.adjmat[i, k] = 1
                Loop = True

        for (i, j, k, l) in Kite:
            if ((j, l, k) in cg_new.definite_UC or (k, l, j) in cg_new.definite_UC) \
                    and ((j, i, k) in cg_new.definite_non_UC or (k, i, j) in cg_new.definite_non_UC) \
                    and cg_new.isUndirected(i, l):
                cg_new.adjmat[i, l] = 1
                Loop = True

    return cg_new

#######################################################################################################################
