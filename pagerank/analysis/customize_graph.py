import matplotlib.pyplot as plt
import random

from pagerank.accelerate_gd import AccelerateGD
from pagerank.accelerate_gd_np import AccelerateGDNumpy
from pagerank.config import Config
from pagerank.proximal_gd import ProximalGD
from pagerank.random_cd import RandomCD

from nzeros import plot_nzeros
from fvalues import plot_fvalues

# function to generate random edges, stupid implementation.
def random_edges(ref_nodes, num_nodes, num_edges, max_try = 1000):
    edge_list = set()
    visited = [node for node in ref_nodes]
    cnt = 0
    while len(edge_list) < num_edges and len(edge_list) < num_nodes * num_nodes and cnt < max_try:
        cnt += 1
        u = random.choice(visited)
        v = random.randint(0, num_nodes - 1)

        if v not in visited:
            visited.append(v)
        edge_list.add((u, v))
    return list(edge_list)

if __name__ == "__main__":
    config_file = 'config.yaml'
    config = Config(config_file)
    config.max_iter = 20

    edge_list = [
        (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (6, 8), (7, 10), (5, 7)
    ]
    config.rho = 1e-6
    config.alpha = 0.5
    
    '''
    edge_list = []
    num_nodes = 100
    num_edges = 100
    edge_list = random_edges(config.ref_nodes, num_nodes, num_edges)
    '''

    legendsize = 16
    xscale = 'log'
    solver = AccelerateGD()
    solver.build_graph(edge_list)
    plot_nzeros(solver, config = config, linestyle = 'solid', color = 'red', xscale = xscale)
    
    solver = ProximalGD()
    solver.build_graph(edge_list)
    plot_nzeros(solver, config = config, linestyle = 'dashdot', color = 'black', xscale = xscale)
    
    plt.legend(prop = {'size': legendsize})
    plt.show()