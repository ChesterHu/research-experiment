import numpy as np
import localgraphclustering as lgc
import matplotlib.pyplot as plt

from pagerank.bounded_accelerated_proximal_gradient_descent import BoundedAcceleratedProximalGradientDescent

if __name__ == "__main__":
    graph_name = '../data/ppi_mips.graphml'
    graph_type = 'graphml'

    alpha = 0.15
    epsilon = 1e-6
    rho = 1e-4
    graph = lgc.GraphLocal(graph_name, graph_type)
    seed_nodes = [0]

    solver = BoundedAcceleratedProximalGradientDescent(graph, seed_nodes = seed_nodes, alpha = alpha, epsilon = epsilon, rho = rho)
    solver.set_max_iter(100)
    solver.solve()