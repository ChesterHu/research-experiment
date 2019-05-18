"""
This script produce plot for comparing function values in methods
"""
import matplotlib.pyplot as plt

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_gd import AccelerateGD
from pagerank.proximal_gd import ProximalGD
from pagerank.config import Config

def plot_fvalues(solver, config, linestyle = 'solid', color = 'red'):
    solver.load_graph(config.graph_file, config.graph_type)
    _, fvalues, _, _ = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    iterations = [i for i in range(1, len(fvalues) + 1)]
    plt.plot(iterations, fvalues, label = str(solver), linestyle = linestyle, linewidth = 3, color = color)

if __name__ == "__main__":

    # load configs
    config_file = 'config.yaml'
    config = Config(config_file)

    # plot
    fontsize = 20
    legendsize = 20

    plt.subplot(1, 2, 1)
    plot_fvalues(AccelerateCD(), config, linestyle = 'solid', color = 'red')
    plot_fvalues(RandomCD(), config, linestyle = 'dotted', color = 'black')
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('function vales', fontsize = fontsize)
    plt.xscale('log')

    plt.subplot(1, 2, 2)
    config.max_iter = 100
    plot_fvalues(ProximalGD(), config, linestyle = 'dashed', color = 'green')
    plot_fvalues(AccelerateGD(), config, linestyle = 'solid', color = 'purple')
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('function vales', fontsize = fontsize)
    plt.xscale('log')
    
    plt.show()