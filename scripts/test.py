import os
import time
import numpy as np
import matplotlib.pyplot as plt

import localgraphclustering as lgc

"""
print("loading graph...")
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
t = time.clock()
nodes, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")
print(f'non-randomized algorithm uses time: {time.clock() - t}')
print(nodes)

g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
t = time.clock()
nodes, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg-rand")
print(f'randomized algorithm uses time: {time.clock() - t}')
print(nodes)

# test warm start
print('test warm start')
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
nodes1, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")
# start after non-randomized method
nodes2, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg-rand")
nodes3, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")

print(f'First converge:\n{nodes1}')
print(f'Second converge:\n{nodes2}')
print(f'Third converge:\n{nodes3}')
"""

def test_time(test_method, epsilons, graph_name):
    """
    Test time for method
    """
    alpha = 0.15
    rho = 0.0001
    ref_node = [x for x in range(10)]
    times = []
    g = lgc.GraphLocal(f"../../LocalGraphClustering/notebooks/datasets/{graph_name}.graphml", "graphml")
    for eps in epsilons:
        t = time.clock()
        lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = test_method)
        times.append((time.clock() - t) * 1000)
    return times

def plot_run_time(epsilons, graph_name, rand_time, norm_time):
    plt.title(f'Running time on graph {graph_name}')
    plt.semilogx(epsilons, rand_time, label='random')
    plt.semilogx(epsilons, norm_time, label='non-random')
    
    plt.xlabel('Epsilon')
    plt.ylabel('Time (ms)')

    plt.legend()
    fname = f'../figures/run-time-eps-{graph_name}.png'
    if os.path.isfile(fname):
        print(f'file: {fname} already exists')
    plt.savefig(fname)
    plt.close()

def compare_result(graph_name):
    alpha = 0.15
    eps = 1e-6
    rho = 0.0001
    ref_node = [x for x in range(10)]
    g = lgc.GraphLocal(f"../../LocalGraphClustering/notebooks/datasets/{graph_name}.graphml", "graphml")
    nodes, _ = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = 'l1reg')
    print(f'results from non-random method:\n{nodes}')
    nodes, _ = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = 'l1reg-rand')
    print(f'results from non-random method:\n{nodes}')

if __name__ == "__main__":
    graph_name = 'JohnsHopkins'
    epsilons = np.logspace(-1, -6, 30)
    rand_time = test_time("l1reg-rand", epsilons, graph_name)
    norm_time = test_time("l1reg", epsilons, graph_name)
    plot_run_time(epsilons, graph_name, rand_time, norm_time)
    compare_result(graph_name)
