import numpy as np

from coordinate_descent import CoordinateDescent

class AccelerateCD(CoordinateDescent):
    def __init__(self):
        self.g = None

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        pass

    def update_gradients(self, node, alpha, rho, q, gradients, candidates):
        pass

    def update_candidates(self, node, alpha, rho, q, gradients, candidates):
        pass


if __name__ == "__main__":
    import os
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)

    alpha = 0.15
    rho = 1e-4
    epsilon = 1e-4
    max_iter = 1e6
    solver = AccelerateCD()
    