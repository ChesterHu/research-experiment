"""
This script produce plot for comparing function values in both iteration.
"""

import matplotlib.pyplot as plt

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_cd_fast import AccelerateCDFast
from pagerank.test_config import TestConfig

def plot_fvalues(solver, config, linestyle = 'solid', color = 'red'):
    solver.load_graph(config.graph_file, graph_type)
    _, fvalues, _, _ = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    iterations = [i for i in range(1, len(fvalues) + 1)]
    plt.plot(iterations, fvalues, label = str(solver), linestyle = linestyle, linewidth = 3, color = color)

if __name__ == "__main__":

    # experiment parameters
    ref_nodes = [4]
    alpha = 0.05
    rho = 1e-4
    epsilon = 1e-8
    max_iter = 1000
    graph_type = 'graphml'
    graph_file = 'ppi_mips.graphml'
    config = TestConfig(ref_nodes, alpha, rho, epsilon, max_iter, graph_file, graph_type)

    # solve
    plot_fvalues(AccelerateCD(), config, linestyle = 'solid', color = 'red')
    plot_fvalues(AccelerateCDFast(), config, linestyle = 'dotted', color = 'black')
    plot_fvalues(RandomCD(), config, linestyle = 'dashed', color = 'green')

    # plot
    fontsize = 20
    legendsize = 20
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('function vales', fontsize = fontsize)
    plt.xscale('log')
    plt.show()