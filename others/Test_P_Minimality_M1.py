#######################################################################################################################
from GraphClass import tetradToCausalGraph
from itertools import combinations
from Helper import powerset, listMinus
from GenerateDAG import dagsGenerator, nxGraphToPattern
from Test_CMC import CMCTester
import numpy as np
import networkx as nx
import time
#######################################################################################################################

def pMinimalityM1Tester(true_cg, data, test_name, alpha, **kwargs):
    """Test Pearl Minimality
    :param true_cg: the true CausalGraph object (use tetradToCausalGraph)
    :param data: data set (numpy ndarray)
    :param test_name: name of the independence test being used (string)
    :param alpha: a desired significance levels in (0, 1) (float)
    :return:
    1. True if P-Minimality is satisfied and False otherwise (using M1)
    2. CI_facts: a list of conditional independence relations
    3. CD_facts: a list of conditional dependence relations
    """
    true_cg.data = data
    true_cg.setTestName(test_name)
    true_cg.corr_mat = np.corrcoef(data, rowvar=False) if test_name == "Fisher_Z" else []

    # First, check whether G* is Markov and obtain I(G*)

    if "CMC_result" in kwargs:
        CMC = kwargs["CMC_result"][0]
        I_G_star = kwargs["CMC_result"][1]
    else:
        CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)

    if not CMC:
        return False, [], []
    else:
        no_of_nodes = data.shape[1]
        range_of_nodes = range(no_of_nodes)
        CI_facts = []   # Save facts of conditional independence for convenience
        CD_facts = []   # Save facts of conditional dependence for convenience

        # We try to construct a DAG G' where
        # 1. I(G*) is a proper subset of I(G')
        # 2. I(G') is a subset of I(P) (i.e., G' is Markov)

        set_of_true_adj = set([(i, j) for (i, j) in true_cg.findAdj() if i < j])
        no_of_edges_prime = len(set_of_true_adj) - 1
        # Lemma: for any pair of DAGs G0 and G1, if I(G0) is a proper subset of I(G1),
        # the adjacencies of G1 is a proper subset of the adjacencies of G0.

        while no_of_edges_prime >= 0:

            patterns_list = []

            for g_prime in dagsGenerator(no_of_nodes, no_of_edges_prime):

                # Due to Lemma, we only search for DAGs with a subset of adjacencies
                e = g_prime.edges
                set_of_prime_adj = set([(i, j) for (i, j) in e if i < j] + [(j, i) for (i, j) in e if i > j])
                if not set_of_prime_adj.issubset(set_of_true_adj):
                    continue # Look for next g_prime
                else:
                    p = nxGraphToPattern(g_prime)

                    if len(patterns_list) == 0:
                        existing_pattern = False
                    else:
                        existing_pattern = False
                        for pattern_old in patterns_list:
                            if np.array_equal(p, pattern_old, equal_nan = True):
                                existing_pattern = True
                                break

                    if existing_pattern:
                        continue
                    else:
                        patterns_list.append(p)

                    # Next, we check I(G*) is a subset of I(G')
                    # If yes, I(G') is a proper subset of I(G*) because G' has fewer edges than G*
                    # Otherwise, we move on to the next G'
                    next_prime = False
                    for CI in I_G_star:
                        if not nx.d_separated(g_prime, {CI[0]}, {CI[1]}, CI[2]):
                            next_prime = True
                            break
                    if next_prime:
                        continue # Look for next g_prime
                    else:

                        # Finally, we check if I(G') is a subset of I(P) (i.e., G' is Markov)
                        possible_adj = list(combinations(range_of_nodes, 2))
                        prime_nonadj = [(i, j) for (i, j) in possible_adj if (i, j) not in set_of_prime_adj]
                        prime_CMC = True

                        for (i, j) in prime_nonadj:
                            if not prime_CMC:
                                break
                            remaining_nodes = listMinus(range_of_nodes, [i, j])
                            cond_sets = powerset(remaining_nodes)
                            for S in cond_sets:
                                if [i, j, S] in I_G_star:
                                    continue # Check next CI if it has been already checked in I(G*)
                                elif nx.d_separated(g_prime, {i}, {j}, S):
                                    if [i, j, S] in CI_facts:
                                        continue
                                    elif [i, j, S] in CD_facts:
                                        prime_CMC = False
                                        break
                                    else:
                                        p = true_cg.ci_test(i, j, S)
                                        if p > alpha:
                                            CI_facts.append([i, j, S])
                                            continue
                                        else:
                                            CD_facts.append([i, j, S])
                                            prime_CMC = False
                                            break

                        if not prime_CMC:
                            continue # if G' is not Markov, look for the next G'
                        else:
                            return False, CI_facts, CD_facts # G' is found and thus G* is not p-minimal

            no_of_edges_prime += -1

        return True, CI_facts, CD_facts # the while-loop terminates if no such G' exists

######################################################################################################################

if __name__ == "__main__":
    alpha = 0.01
    test_name = "Fisher_Z"
    truth_path = "simulational_studies/Example_4.3.1_true_DAG.txt"
    data_path = "simulational_studies/Example_4.3.1_data.txt"
    true_cg = tetradToCausalGraph(truth_path)
    data = np.loadtxt(data_path, skiprows=1)
    CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)
    start = time.time()
    pMinimality, _, _ = pMinimalityM1Tester(true_cg, data, test_name, alpha, CMC_result = [CMC, I_G_star])
    end = time.time()
    print(f"Time elapsed: {round(end - start, 2)} seconds; Pearl-Minimality: {pMinimality}")

######################################################################################################################