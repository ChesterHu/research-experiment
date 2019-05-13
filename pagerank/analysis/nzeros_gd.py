"""
Test non-zero nodes in gradient descent and accelerated gradient descent
"""
import matplotlib.pyplot as plt

from nzeros import plot_nzeros
from fvalues import plot_fvalues

from pagerank.accelerate_gd import AccelerateGD
from pagerank.proximal_gd import ProximalGD
from pagerank.test_config import TestConfig

if __name__ == "__main__":
    
    # experiment parameters
    ref_nodes = [_ for _ in range(10)]
    alpha = 0.05
    rho = 1e-4
    epsilon = 1e-8
    max_iter = 20
    graph_type = 'graphml'
    graph_file = 'ppi_mips.graphml'
    config = TestConfig(ref_nodes, alpha, rho, epsilon, max_iter, graph_file, graph_type)

    # solve
    plot_nzeros(AccelerateGD(), config, linestyle = 'solid', color = 'red')
    plot_nzeros(ProximalGD(), config, linestyle = 'dashed', color = 'black')

    # plot
    fontsize = 20
    legendsize = 20
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('number of non-zero nodes', fontsize = fontsize)
    plt.show()