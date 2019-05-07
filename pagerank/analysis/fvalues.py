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
    _, fvalues, __ = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    iterations = [i for i in range(1, len(fvalues) + 1)]
    plt.plot(iterations, fvalues, label = str(solver), linestyle = linestyle, color = color)

if __name__ == "__main__":

    # experiment parameters
    ref_nodes = [4]
    alpha = 0.1
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
    """
    # solve accelerated method
    accel_solver = AccelerateCD()
    accel_solver.load_graph(graph_file, graph_type)
    accel_q, accel_fvalues, accel_nzeros = accel_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)
    # solve non-accelerated method
    rand_solver = RandomCD()
    rand_solver.load_graph(graph_file, graph_type)
    rand_q, rand_fvalues, rand_nzeros = rand_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)
    # solve accelerated method fast version
    fast_solver = AccelerateCDFast()
    fast_solver.load_graph(graph_file, graph_type)
    fast_q, fast_fvalues, fast_nzeros = fast_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

    # plot function value in both methods
    plt.plot([_ + 1 for _ in range(len(accel_fvalues))], accel_fvalues, label = str(accel_solver), linestyle = 'solid', linewidth = 3, color = 'red')
    plt.plot([_ + 1 for _ in range(len(rand_fvalues))], rand_fvalues, label = str(rand_solver), linestyle = 'dashed', linewidth = 3, color = 'black')
    plt.plot([_ + 1 for _ in range(len(fast_fvalues))], fast_fvalues, label = str(fast_solver), linestyle = 'dotted', linewidth = 3, color = 'green')

    # plot settings
    plt.xscale('log')
    plt.legend(prop={'size': 18})
    plt.xlabel('iterations', fontsize = 20)
    plt.ylabel('function value', fontsize = 20)
    plt.show()
    """