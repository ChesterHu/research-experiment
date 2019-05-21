"""
This script produce plot for comparing number of non-zero nodes in methods. 
The accelerated method can touch more non-zero nodes than non-accelerated method, but it uses less iterations to converge.
"""
import matplotlib.pyplot as plt

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_gd import AccelerateGD
from pagerank.proximal_gd import ProximalGD
from pagerank.config import Config

def plot_nzeros(solver, **kwargs):
    color = kwargs.get('color', 'red')
    config = kwargs.get('config', None)
    fontsize = kwargs.get('fontsize', 20)
    linestyle = kwargs.get('linestyle', 'solid')
    linewidth = kwargs.get('linewidth', 3)
    xscale = kwargs.get('xscale', 'linear')

    # load graph if it's not set
    if config.graph_file:
        solver.load_graph(config.graph_file, config.graph_type)
    elif not solver.g:
        raise ValueError('graph is empty')
    _, _, nzeros, _ = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    optimal_nzeros = nzeros[-1]
    
    plt.xlabel('iterations', fontsize = fontsize)
    plt.ylabel('number of non-zero nodes', fontsize = fontsize)
    plt.xscale(xscale)

    plt.plot(nzeros, label = str(solver), linestyle = linestyle, linewidth = linewidth, color = color)
    plt.axhline(y = optimal_nzeros, linestyle = 'dashdot', linewidth = 3, color = 'blue')


if __name__ == "__main__":

    # load configs
    config_file = 'config.yaml'
    config = Config(config_file)
    legendsize = 20

    # coordinate descent
    plt.subplot(1, 2, 1)
    plot_nzeros(AccelerateCD(), linestyle = 'solid', color = 'red', config = config)
    plot_nzeros(RandomCD(), linestyle = 'dashed', color = 'black', config = config)
    plt.legend(prop = {'size': legendsize})

    # gradient descent
    plt.subplot(1, 2, 2)
    config.max_iter = 50 # gradient descent uses less iterations
    plot_nzeros(AccelerateGD(), linestyle = 'solid', color = 'green', config = config)
    plot_nzeros(ProximalGD(), linestyle = 'dashed', color = 'purple', config = config)
    plt.legend(prop = {'size': legendsize})
    plt.show()