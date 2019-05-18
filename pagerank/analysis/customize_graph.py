from pagerank.accelerate_gd_np import AccelerateGDNumpy
from pagerank.proximal_gd import ProximalGD

if __name__ == "__main__":
    edge_list = [
        (0, 0), (0, 1), (0, 3),
        (1, 1), (1, 3),
        (2, 3),
        (3, 3)
    ]

    ref_nodes = [0, 1]
    alpha = 0.15
    rho = 1e-6
    epsilon = 1e-8
    max_iter = 1000

    solver = AccelerateGDNumpy()
    solver.build_graph(edge_list)
    
    q, fvalues, nzeros, times = solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)