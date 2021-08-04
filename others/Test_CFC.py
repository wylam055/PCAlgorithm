#######################################################################################################################
from GraphClass import tetradToCausalGraph
from itertools import combinations
from Test_CMC import CMCTester
from Helper import powerset, listMinus
from copy import deepcopy
import time
import numpy as np
#######################################################################################################################


def faithfulnessTester(true_cg, data, test_name, alpha, **kwargs):
    """Test CFC, adj_faithfulness, and ori_faithfulness
    :param true_cg: the true CausalGraph object (use tetradToCausalGraph)
    :param data: data set (numpy ndarray)
    :param test_name: name of the independence test being used (string)
    :param alpha: a desired significance levels in (0, 1) (float)
    :return:
    1. CFC: True if CFC is satisfied and False otherwise
    2. adj_faithful: True if adj-faithfulness is satisfied and False otherwise
    3. ori_faithful: True if ori-faithfulness is satisfied and False otherwise
    """
    cg = true_cg
    cg.data = data
    cg.setTestName(test_name)
    cg.corr_mat = np.corrcoef(data, rowvar=False) if test_name == "Fisher_Z" else []

    if "CMC_result" in kwargs:
        CMC = kwargs["CMC_result"][0]
        I_G_star = kwargs["CMC_result"][1]
    else:
        CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)

    if not CMC:
        return False, False, False
    else:
        CI_facts = kwargs["CI_facts"] + I_G_star if "CI_facts" in kwargs else deepcopy(I_G_star)
        CD_facts = kwargs["CD_facts"] if "CD_facts" in kwargs else deepcopy([])

    range_of_nodes = range(data.shape[1])
    adj = [(i, j) for (i, j) in cg.findAdj() if i < j]
    UT = [(i, j, k) for (i, j, k) in cg.findUnshieldedTriples() if i < k]

    CFC = True
    adj_faithful = True
    ori_faithful = True

    for (i, j) in combinations(range_of_nodes, 2):
        remaining_nodes = listMinus(range_of_nodes, [i, j])
        cond_sets = powerset(remaining_nodes)
        if (i, j) in adj and adj_faithful:
            for S in cond_sets:
                if [i, j, S] in CI_facts:
                    adj_faithful = False
                    CFC = False
                elif [i, j, S] in CD_facts:
                    continue
                else:
                    p = cg.ci_test(i, j, S)
                    if p > alpha:
                        adj_faithful = False
                        CFC = False
        else:
            UT_ij = [(x, y, z) for (x, y, z) in UT if x == i and z == j]
            if len(UT_ij) != 0 and ori_faithful:
                for (x, y, z) in UT_ij:
                    if cg.isCollider(x, y, z):
                        for S in [S_sets for S_sets in cond_sets if y in S_sets]:
                            if [i, j, S] in CI_facts:
                                ori_faithful = False
                                CFC = False
                            elif [i, j, S] in CD_facts:
                                continue
                            else:
                                p = cg.ci_test(i, j, S)
                                if p > alpha:
                                    ori_faithful = False
                                    CFC = False
                    else:
                        for S in [S_sets for S_sets in cond_sets if y not in S_sets]:
                            if [i, j, S] in CI_facts:
                                ori_faithful = False
                                CFC = False
                            elif [i, j, S] in CD_facts:
                                continue
                            else:
                                p = cg.ci_test(i, j, S)
                                if p > alpha:
                                    ori_faithful = False
                                    CFC = False
            elif CFC:
                for S in cond_sets:
                    if not cg.isDSep(i, j, S):
                        if [i, j, S] in CI_facts:
                            CFC = False
                        elif [i, j, S] in CD_facts:
                            continue
                        else:
                            p = cg.ci_test(i, j, S)
                            if p > alpha:
                                CFC = False

    return CFC, adj_faithful, ori_faithful

#######################################################################################################################

if __name__ == "__main__":

    alpha = 0.01
    test_name = "Fisher_Z"
    truth_path = "simulational_studies/Example_4.3.1_true_DAG.txt"
    data_path = "simulational_studies/Example_4.3.1_data.txt"
    true_cg = tetradToCausalGraph(truth_path)
    data = np.loadtxt(data_path, skiprows=1)
    CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)
    start = time.time()
    CFC, adj_faithful, ori_faithful = faithfulnessTester(true_cg, data, test_name, alpha, CMC_result = [CMC, I_G_star])
    end = time.time()
    print(f"Time elapsed: {round(end - start, 2)} seconds; CFC: {CFC}, ",
          f"adj-faithful: {adj_faithful}, ", f"ori-faithful: {ori_faithful}")

#######################################################################################################################