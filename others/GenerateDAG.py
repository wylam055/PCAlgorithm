#######################################################################################################################
from itertools import combinations, product
from Helper import powerset, listMinus, dag2Pattern
import networkx as nx
from networkx import d_separated
import numpy as np
#######################################################################################################################

def dSepRelations(nx_graph):
    nodes = nx_graph.nodes
    edges = list(nx_graph.edges)
    set_of_adj = set([(i, j) for (i, j) in edges if i < j] + [(j, i) for (i, j) in edges if i > j])
    possible_adj = list(combinations(nodes, 2))
    nonadj = [(i, j) for (i, j) in possible_adj if (i, j) not in set_of_adj]
    for (i, j) in nonadj:
        remaining_nodes = listMinus(nodes, [i, j])
        cond_sets = powerset(remaining_nodes)
        for S in cond_sets:
            if d_separated(nx_graph, {i}, {j}, S):
                yield [i, j, S]

#######################################################################################################################

def nxGraphToPattern(nx_graph):
    no_of_var = len(nx_graph.nodes)
    assert no_of_var > 0
    adjmat = np.zeros((no_of_var, no_of_var))
    np.fill_diagonal(adjmat, None)
    adjmat[adjmat == 0] = -1
    for (i, j) in nx_graph.edges:
        adjmat[i, j] = 1
        adjmat[j, i] = 0
    pattern = dag2Pattern(adjmat)
    return pattern

#######################################################################################################################

def dagsGenerator(no_of_nodes, no_of_edges):
    """Create zero_one_lists list of DAGs (nx.DiGraph objects) with assigned number of nodes and edges"""
    if (no_of_nodes * (no_of_nodes - 1))/2 < no_of_edges:
        return []
    else:
        range_of_nodes = range(no_of_nodes)

        list_of_possible_adj = list(combinations(range_of_nodes, 2))
        list_of_adj_sets = list(combinations(list_of_possible_adj, no_of_edges))
        for adj_set in list_of_adj_sets:
            bool_combinations = list(product([True, False], repeat=no_of_edges))

            for bool_comb in bool_combinations:
                add_directed_edges = []
                for k in range(len(bool_comb)):
                    if bool_comb[k]:
                        add_directed_edges.append(adj_set[k])
                    else:
                        add_directed_edges.append((adj_set[k][1], adj_set[k][0]))

                g = nx.DiGraph()
                g.add_nodes_from(range_of_nodes)
                for (i, j) in add_directed_edges:
                    g.add_edge(i, j, color='b')
                if nx.is_directed_acyclic_graph(g):
                    yield g

#######################################################################################################################