#######################################################################################################################
from GraphClass import tetradToCausalGraph, toPattern, CausalGraph
from GenerateDAG import dagsGenerator, dSepRelations, nxGraphToPattern
from Test_CMC import CMCTester
from itertools import permutations
import numpy as np
import networkx as nx
from itertools import combinations
import time
from copy import deepcopy
#######################################################################################################################

#######################################################################################################################

def causal_ordering(no_of_nodes):
    for order in permutations(range(no_of_nodes), no_of_nodes):
        yield order

#######################################################################################################################

def pMinM2FruTester(true_cg, data, test_name, alpha, **kwargs):
    """Test Pearl Minimality, frugality, and unique frugality
    :param true_cg: the true CausalGraph object (use tetradToCausalGraph)
    :param data: data set (numpy ndarray)
    :param test_name: name of the independence test being used (string)
    :param alpha: a desired significance levels in (0, 1) (float)
    :return:
    1. P_minimal: True if P-Minimality is satisfied and False otherwise (using M2)
    2. Frugal: True if frugality is satisfied and False otherwise
    3. u_frugal: True if u-frugality is satisfied and False otherwise
    4. CI_facts: a list of conditional independence relations
    5. CD_facts: a list of conditional dependence relations
    """
    true_cg.data = data
    true_cg.setTestName(test_name)
    true_cg.corr_mat = np.corrcoef(data, rowvar=False) if test_name == "Fisher_Z" else []

    if "CMC_result" in kwargs:
        CMC = kwargs["CMC_result"][0]
        I_G_star = kwargs["CMC_result"][1]
    else:
        CMC, I_G_star = CMCTester(true_cg, data, test_name, alpha)

    if not CMC:
        return [False, False, False, [], []]
    else:
        CI_facts = kwargs["CI_facts"] + I_G_star if "CI_facts" in kwargs else deepcopy(I_G_star)
        CD_facts = kwargs["CD_facts"] if "CD_facts" in kwargs else deepcopy([])

        no_of_nodes = data.shape[1]
        nodes = range(no_of_nodes)
        no_of_true_edges = len(true_cg.findFullyDirected())
        true_adj = set([(i, j) for (i, j) in true_cg.findAdj() if i < j])

        def DAG_construct(pi):
            pi_edges = []
            for (j, k) in combinations(nodes, 2): # j < k guaranteed
                pi_j = pi[j]
                pi_k = pi[k]
                pi_cond_set = pi[0:j] + pi[j+1:k]
                if (pi_j, pi_k, pi_cond_set) in CI_facts:
                    continue
                elif (pi_j, pi_k, pi_cond_set) in CD_facts:
                    pi_edges.append((pi_j, pi_k))
                    continue
                else:
                    p = true_cg.ci_test(pi_j, pi_k, pi_cond_set)
                    if p > alpha:
                        CI_facts.append([pi_j, pi_k, pi_cond_set])
                        continue
                    else:
                        CD_facts.append([pi_j, pi_k, pi_cond_set])
                        pi_edges.append((pi_j, pi_k))
                        continue
            return pi_edges

        def patternEdges(cg):
            pattern = toPattern(cg, checkDAG=False)
            pattern_directed = set(pattern.findFullyDirected())
            pattern_undirected = set([(i, j) for (i, j) in pattern.findUndirected() if i < j])
            return pattern_directed, pattern_undirected

        possibly_max_frugal_DAGs = []

        P_minimal = True
        Frugal = True

        for pi in causal_ordering(no_of_nodes):
            if not P_minimal:
                return [False, False, False, CI_facts, CD_facts]
            pi_edges = DAG_construct(pi)
            # We know that the set of pi_edges will constitute a SGS-minimal Markovian DAG G_pi.

            # Case 1: When |E(G_pi)| > |E(G*)|, G_pi cannot prove the non-P-minimality or non-frugality of G*
            if len(pi_edges) > no_of_true_edges:
                continue

            # Case 2: When |E(G_pi)| = |E(G*)|, G_pi is possibly a maximally frugal DAG
            elif len(pi_edges) == no_of_true_edges:
                if Frugal:
                    possibly_max_frugal_DAGs.append(pi_edges)

            # Case 3: When |E(G_pi) < |E(G*)|, frugality fails but we still need to know whether G* is P-minimal
            else:
                Frugal = False

                # If I(G*) is a proper subset of I(G_pi), adj(G_pi) is a proper subset of adj(G*)
                pi_adj = set([(i, j) for (i, j) in pi_edges if i < j] + [(j, i) for (i, j) in pi_edges if i > j])
                if not pi_adj.issubset(true_adj): # If subset, then proper subset (because of Case 3)
                    continue # Look for next G_pi
                else:
                    # Next, we check I(G*) is a subset of I(G_pi).
                    # We construct the nx_graph object for G_pi to check d-separation.
                    pi_nx_graph = nx.DiGraph()
                    pi_nx_graph.add_nodes_from(nodes)
                    pi_nx_graph.add_edges_from(pi_edges)
                    next_pi = False
                    for CI in I_G_star:
                        if not nx.d_separated(pi_nx_graph, {CI[0]}, {CI[1]}, CI[2]):
                            next_pi = True
                            break
                    if next_pi:
                        continue # Look for next pi
                    else:
                        P_minimal = False

        if not P_minimal:
            return [False, False, False, CI_facts, CD_facts]
        elif not Frugal:
            return [True, False, False, CI_facts, CD_facts]
        else:
            true_pattern_directed, true_pattern_undirected = patternEdges(true_cg)
            for pi_edges in possibly_max_frugal_DAGs:
                est_cg = CausalGraph(no_of_nodes)
                est_cg.adjmat[est_cg.adjmat == 0] = -1 # Initiate an empty graph
                for (j, k) in pi_edges:
                    est_cg.addDirectedEdge(j, k)
                est_pattern_directed, est_pattern_undirected = patternEdges(est_cg)
                if est_pattern_directed != true_pattern_directed or est_pattern_undirected != true_pattern_undirected:
                    return [True, True, False, CI_facts, CD_facts]
                else:
                    continue
            return [True, True, True, CI_facts, CD_facts]

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
    result_list = pMinM2FruTester(true_cg, data, test_name, alpha, CMC_result = [CMC, I_G_star])
    end = time.time()
    print(f"Time elapsed: {round(end - start, 2)} seconds; "
          f"P-Minimality: {result_list[0]}; Frugality: {result_list[1]}; u-Frugality: {result_list[2]}")

#######################################################################################################################