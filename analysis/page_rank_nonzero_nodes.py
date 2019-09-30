import numpy as np
import localgraphclustering as lgc
import matplotlib.pyplot as plt

from pagerank.proximal_gradient_descent import ProximalGradientDescent
from pagerank.accelerated_proximal_gradient_descent import AcceleratedProximalGradientDescent
from pagerank.fast_iterative_shrinkage_algorithm import FastIterativeShrinkageAlgorithm

def get_number_of_non_zeros(solver, iterations = 50):
    non_zeros = []
    for i in range(1, iterations + 1):
        solver.set_max_iter(i)
        q = solver.solve()
        non_zeros.append(len(np.nonzero(q)[0]))
    return non_zeros

if __name__ == "__main__":
    graph_name = '../data/ppi_mips.graphml'
    graph_type = 'graphml'

    alpha = 0.15
    epsilon = 1e-6
    rho = 1e-4
    graph = lgc.GraphLocal(graph_name, graph_type)
    seed_nodes = [0]

    algorithms = [ProximalGradientDescent, AcceleratedProximalGradientDescent, FastIterativeShrinkageAlgorithm]
    colors = ['black', 'red', 'blue']
    linestyles = ['-.', '-', ':']

    for algorithm, color, linestyle in zip(algorithms, colors, linestyles):
        solver = algorithm(graph, seed_nodes = seed_nodes, alpha = alpha, epsilon = epsilon, rho = rho)
        non_zeros = get_number_of_non_zeros(solver)
        plt.plot(non_zeros, label = str(solver), color = color, linestyle = linestyle)

    plt.xlabel('iteration')
    plt.ylabel('number of non zero nodes')
    plt.legend()
    plt.show()