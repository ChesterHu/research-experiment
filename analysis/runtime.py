"""
This script produce plot for comparing running time 
"""

import matplotlib.pyplot as plt

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_cd_fast import AccelerateCDFast
from pagerank.accelerate_gd import AccelerateGD
from pagerank.accelerate_gd_np import AccelerateGDNumpy
from pagerank.config import Config


def plot_runtime(solver, config, linestyle = 'solid', color = 'red'):
    solver.load_graph(config.graph_file, config.graph_type)
    _, fvalues, _, times = solver.solve(config.ref_nodes, config.alpha, config.rho, config.epsilon, config.max_iter)
    plt.plot([t + 1 for t in times], fvalues, label = str(solver), linestyle = linestyle, linewidth = 3, color = color)

if __name__ == "__main__":
    
    # load configs
    config_file = 'config.yaml'
    config = Config(config_file)

    # solve
    plot_runtime(AccelerateGD(), config, linestyle = 'dotted', color = 'blue')
    plot_runtime(AccelerateGDNumpy(), config, linestyle = 'dashdot', color = 'purple')
    
    plot_runtime(RandomCD(), config, linestyle = 'dashed', color = 'green')
    plot_runtime(AccelerateCDFast(), config, linestyle = 'solid', color = 'red')
    

    fontsize = 20
    legendsize = 20
    plt.legend(prop = {'size': legendsize})
    plt.xlabel('time(ms)', fontsize = fontsize)
    plt.ylabel('function value', fontsize = fontsize)
    plt.title('running time')
    # plt.yscale('symlog')
    plt.xscale('log')
    plt.show()