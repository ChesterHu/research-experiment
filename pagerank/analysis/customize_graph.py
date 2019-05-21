import matplotlib.pyplot as plt
import random

from pagerank.accelerate_gd_np import AccelerateGDNumpy
from pagerank.config import Config
from pagerank.proximal_gd import ProximalGD

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
    '''
    edge_list = [
        (0, 1), 
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6)
    ]
    '''
    edge_list = []
    legendsize = 16
    num_nodes = 100
    num_edges = 100
    

    config_file = 'config.yaml'
    config = Config(config_file)
    config.max_iter = 20
    edge_list = random_edges(config.ref_nodes, num_nodes, num_edges)

    solver = AccelerateGDNumpy()
    solver.build_graph(edge_list)
    plot_nzeros(solver, config = config, linestyle = 'solid', color = 'red')
    
    solver = ProximalGD()
    solver.build_graph(edge_list)
    plot_nzeros(solver, config = config, linestyle = 'dashdot', color = 'black')

    plt.legend(prop = {'size': legendsize})
    plt.show()