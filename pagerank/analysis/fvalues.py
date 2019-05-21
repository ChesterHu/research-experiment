"""
This script produce plot for comparing function values in methods
"""
import matplotlib.pyplot as plt

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_gd import AccelerateGD
from pagerank.proximal_gd import ProximalGD
from pagerank.config import Config

def plot_fvalues(solver, **kwargs):
    color = kwargs.get('color', 'red')
    config = kwargs.get('config', None)
    fontsize = kwargs.get('fontsize', 20)
    linestyle = kwargs.get('linestyle', 'solid')
    linewidth = kwargs.get('linewidth', 3)
    xscale = kwargs.get('xscale', 'log')

    if config.graph_file and solver.g is None:
        solver.load_graph(config.graph_file, config.graph_type)
    elif not solver.g:
        raise ValueError('graph is empty')
    _, fvalues, _, _ = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    iterations = [i + 1 for i in range(len(fvalues))]

    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('function value', fontsize = fontsize)
    plt.xscale(xscale)

    plt.plot(iterations, fvalues, label = str(solver), linestyle = linestyle, linewidth = linewidth, color = color)


if __name__ == "__main__":

    # load configs
    config_file = 'config.yaml'
    config = Config(config_file)

    # plot
    legendsize = 20

    plt.subplot(1, 2, 1)
    plot_fvalues(AccelerateCD(), linestyle = 'solid', color = 'red', config = config)
    plot_fvalues(RandomCD(), linestyle = 'dotted', color = 'black', config = config)
    plt.legend(prop = {'size': legendsize})


    plt.subplot(1, 2, 2)
    config.max_iter = 100
    plot_fvalues(ProximalGD(), linestyle = 'dashed', color = 'green', config = config)
    plot_fvalues(AccelerateGD(), linestyle = 'solid', color = 'purple', config = config)
    plt.legend(prop = {'size': legendsize})
    
    plt.show()