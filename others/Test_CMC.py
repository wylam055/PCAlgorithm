#######################################################################################################################
from GraphClass import tetradToCausalGraph
from GenerateDAG import dSepRelations
import numpy as np
import time
#######################################################################################################################


def CMCTester(true_cg, data, test_name, alpha):
    """Test CMC
    :param true_cg: the true CausalGraph object (use tetradToCausalGraph)
    :param data: data set (numpy ndarray)
    :param test_name: name of the independence test being used (string)
    :param alpha: a desired significance levels in (0, 1) (float)
    :return:
    1. True if CMC is satisfied and False otherwise
    2. I_G_star: I(G*) if CMC is true, else []
    """
    cg = true_cg
    cg.data = data
    cg.setTestName(test_name)
    cg.corr_mat = np.corrcoef(data, rowvar=False) if test_name == "Fisher_Z" else []

    CMC = True
    I_G_star = []

    for (i, j, S) in dSepRelations(cg.nx_graph):
        if not CMC:
            break
        I_G_star.append([i, j, S])
        p = cg.ci_test(i, j, S)
        if p <= alpha:
            CMC = False
            break

    if CMC:
        return True, I_G_star
    else:
        return False, []

#######################################################################################################################

if __name__ == "__main__":
    alpha = 0.01
    test_name = "Fisher_Z"
    truth_path = "simulational_studies/Example_4.3.1_true_DAG.txt"
    data_path = "simulational_studies/Example_4.3.1_data.txt"
    true_cg = tetradToCausalGraph(truth_path)
    data = np.loadtxt(data_path, skiprows=1)
    start = time.time()
    CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)
    end = time.time()
    print(f"Time elapsed: {round(end-start, 2)} seconds; CMC: {CMC}")

#######################################################################################################################
