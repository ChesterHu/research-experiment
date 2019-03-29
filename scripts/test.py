import os
import time
import numpy as np
import matplotlib.pyplot as plt

import localgraphclustering as lgc

alpha = 0.05
rho = 0.0001
ref_node = [x for x in range(5)]

def single_test(test_method, epsilon = 1e-14, graph_name = 'JohnsHopkins'):
    global alpha
    global rho
    global ref_node
    g = lgc.GraphLocal(f"../../LocalGraphClustering-1/notebooks/datasets/{graph_name}.graphml", "graphml")
    nodes, probs = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = epsilon, method = test_method)
    print(test_method)
    print(f'\n\nnumber of nodes: {len(nodes)}\nnodes:\n{nodes}\n\nprobs:\n{probs}\n')

def test_time(test_method, epsilons, graph_name):
    """
    Test time for method
    """
    global alpha
    global rho
    global ref_node

    times = []
    g = lgc.GraphLocal(f"../../LocalGraphClustering/notebooks/datasets/{graph_name}.graphml", "graphml")
    for eps in epsilons:
        t = time.clock()
        lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = test_method)
        times.append((time.clock() - t) * 1000)
    return times

def plot_run_time(epsilons, graph_name, rand_time, norm_time):
    global alpha
    global rho

    plt.title(f'Running time on graph {graph_name}')
    plt.semilogx(epsilons, rand_time, label='random')
    plt.semilogx(epsilons, norm_time, label='non-random')
    
    plt.xlabel('Epsilon')
    plt.ylabel('Time (ms)')

    plt.legend()
    fname = f'../figures/run-time-eps-{graph_name}_alpha={alpha}_rho={rho}.png'
    if os.path.isfile(fname):
        print(f'file: {fname} already exists')
    plt.savefig(fname)
    plt.close()

def compare_result(graph_name):
    global alpha
    global rho
    global ref_node

    eps = 1e-6
    g = lgc.GraphLocal(f"../../LocalGraphClustering/notebooks/datasets/{graph_name}.graphml", "graphml")
    nodes, _ = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = 'l1reg')
    print(f'results from non-random method:\n{nodes}')
    nodes, _ = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = 'l1reg-rand')
    print(f'results from random method:\n{nodes}')

if __name__ == "__main__":
    '''
    graph_name = 'JohnsHopkins'
    epsilons = np.logspace(-1, -6, 30)
    rand_time = test_time("l1reg-rand", epsilons, graph_name)
    norm_time = test_time("l1reg", epsilons, graph_name)
    plot_run_time(epsilons, graph_name, rand_time, norm_time)
    compare_result(graph_name)
    '''
    single_test('l1reg-rand-accel')