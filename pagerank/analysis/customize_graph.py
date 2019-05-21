import matplotlib.pyplot as plt

from pagerank.accelerate_gd_np import AccelerateGDNumpy
from pagerank.config import Config
from pagerank.proximal_gd import ProximalGD

from nzeros import plot_nzeros
from fvalues import plot_fvalues

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
    
    for i in range(num_nodes - 1):
        edge_list.append((i, i + 1))

    config_file = 'config.yaml'
    config = Config(config_file)

    solver = AccelerateGDNumpy()
    solver.build_graph(edge_list)
    plot_nzeros(solver, config = config, linestyle = 'solid', color = 'red')
    
    solver = ProximalGD()
    solver.build_graph(edge_list)
    plot_nzeros(solver, config = config, linestyle = 'dashdot', color = 'black')

    plt.legend(prop = {'size': legendsize})
    plt.show()