"""
This script produce plot for comparing number of non-zero nodes in methods. 
The accelerated method can touch more non-zero nodes than non-accelerated method, but it uses less iterations to converge.
"""
import matplotlib.pyplot as plt

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_gd import AccelerateGD
from pagerank.proximal_gd import ProximalGD
from pagerank.test_config import TestConfig

def plot_nzeros(solver, config, linestyle = 'solid', color = 'red'):
    solver.load_graph(config.graph_file, config.graph_type)
    _, _, nzeros, _ = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    iterations = [i for i in range(1, len(nzeros) + 1)]
    plt.plot(iterations, nzeros, label = str(solver), linestyle = linestyle, linewidth = 3, color = color)
    plt.axhline(y = nzeros[-1], linestyle = 'dashdot', linewidth = 3, color = 'blue')

if __name__ == "__main__":

    # experiment parameters
    ref_nodes = [3]
    alpha = 0.05
    rho = 1e-4
    epsilon = 1e-8
    max_iter = 1000
    graph_type = 'graphml'
    graph_file = 'ppi_mips.graphml'
    config = TestConfig(ref_nodes, alpha, rho, epsilon, max_iter, graph_file, graph_type)

    # coordinate descent
    plt.subplot(1, 2, 1)
    plot_nzeros(AccelerateCD(), config, linestyle = 'solid', color = 'red')
    plot_nzeros(RandomCD(), config, linestyle = 'dashed', color = 'black')
    
    fontsize = 20
    legendsize = 20
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('number of non-zero nodes', fontsize = fontsize)
    plt.xscale('log')

    # gradient descent
    plt.subplot(1, 2, 2)
    config.max_iter = 100 # gradient descent uses less iterations
    plot_nzeros(AccelerateGD(), config, linestyle = 'solid', color = 'red')
    plot_nzeros(ProximalGD(), config, linestyle = 'dashed', color = 'black')

    fontsize = 20
    legendsize = 20
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('number of non-zero nodes', fontsize = fontsize)
    plt.xscale('log')

    plt.show()