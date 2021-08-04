#######################################################################################################################
from GraphClass import tetradToCausalGraph
from Test_CMC import CMCTester
from Test_CFC import faithfulnessTester
from Test_P_Minimality_M2_and_Frugality import pMinM2FruTester
import numpy as np
import time
#######################################################################################################################

def razorsTester(truth_path, data_path, test_name, alpha):
    """
    Test a list of razors
    :param truth_path: the path of the .txt file which represent the true DAG in TETRAD format
    :param data_path: the path of the .txt file which contains the data
    :param test_name: name of the independence test being used (string)
    :param alpha: a desired significance levels in (0, 1) (float)
    :return:
    1. CMC: True if CMC is satisfied and False otherwise
    2. P_minimality: True if P-minimality is satisfied and False otherwise (using M2)
    3. frugality: True if frugality is satisfied and False otherwise
    4. u_frugality: True if u-frugality is satisfied and False otherwise
    5. adj_faithful: True if adj-faithfulness is satisfied and False otherwise
    6. ori_faithful: True if ori-faithfulness is satisfied and False otherwise
    7. res_faithful: True if both adj-faithfuless and ori-faithfulness are satisfied and False otherwise
    8. CFC: True if CFC is satisfied and False otherwise
    """
    true_cg = tetradToCausalGraph(truth_path)
    data = np.loadtxt(data_path, skiprows=1)
    start = time.time()
    CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)

    if not CMC:
        end1 = time.time()
        return [int(CMC), 0, 0, 0, 0, 0, 0, 0, round(end1 - start, 2)]
    else:
        firstResults = pMinM2FruTester(true_cg, data, test_name, alpha, CMC_result = [CMC, I_G_star])
        pMinimal = firstResults[0]
        Frugal = firstResults[1]
        uFrugal = firstResults[2]
        CI_facts = firstResults[3]
        CD_facts = firstResults[4]
        CFC, adj_faithful, ori_faithful = faithfulnessTester(true_cg, data, test_name, alpha,
                                                             CI_facts=CI_facts, CD_facts=CD_facts)
        end2 = time.time()
        return [int(CMC), int(pMinimal), int(Frugal), int(uFrugal),
                int(adj_faithful), int(ori_faithful), int(adj_faithful and ori_faithful), int(CFC),
                round(end2- start, 2)]

######################################################################################################################

if __name__ == "__main__":
    alpha = 0.01
    test_name = "Fisher_Z"
    truth_path = "simulational_studies/Example_4.3.1_true_DAG.txt"
    data_path = "simulational_studies/Example_4.3.1_data.txt"
    results = razorsTester(truth_path, data_path, test_name, alpha)
    print(f"Time elapsed: {results[-1]} seconds; Results: {results[0:-1]}")
######################################################################################################################
