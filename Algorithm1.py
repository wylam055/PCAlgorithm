#######################################################################################################################
import numpy as np
from GraphClass import CausalGraph
from Helper import appendValue
from itertools import permutations, combinations
#######################################################################################################################


def skeletonDiscovery(data, alpha, test_name, stable=True):
    """Perform skeleton discovery
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param test_name: name of the independence test being used
           - "Fisher_Z": Fisher's Z conditional independence test
           - "Chi_sq": Chi-squared conditional independence test
           - "G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """
    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_name in ["Fisher_Z", "Chi_sq", "G_sq"]

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.data = data
    cg.setTestName(test_name)
    cg.corr_mat = np.corrcoef(data, rowvar=False) if test_name == "Fisher_Z" else []

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.maxDegree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            cg.adjmat[x, y] = -1
                            cg.adjmat[y, x] = -1
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            appendValue(cg.sepset, x, y, S)
                            appendValue(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            cg.adjmat[x, y] = -1

    return cg
#######################################################################################################################
