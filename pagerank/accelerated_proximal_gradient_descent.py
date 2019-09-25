import numpy as np
from math import sqrt

from pagerank.page_rank_solver import PageRankSolver

class AcceleratedProximalGradientDescent(PageRankSolver):
    """
    Implementation of accelerated proximal gradient descent to solve problem:
        min F(q) = f(q) + g(q)
    
    Assumption:
        1. f(q) is a smooth convex function, i.e continuously differentiable with Lipschitz constant L
        2. g(q) is convex and possibly nonsmooth
        3. min F(q) is sovable
    """
    
    def __str__(self):
        return 'accelerated proximal gradient descent'

    def solve(self):
        """
        Solve the problem by accelerated proximal gradient descent
        """
        q = np.zeros(self.graph._num_vertices, dtype = float)
        prev_q = np.zeros(self.graph._num_vertices, dtype = float)
        return self.minimize(q, prev_q)

    def minimize(self, q, prev_q):
        """
        Minimize the objective function by accelerated proximal gradient descent
        """
        return q, prev_q

    def get_beta(self, num_iter):
        if num_iter == 1:
            return 0
        return (1 - sqrt(self.alpha)) / (1 + sqrt(self.alpha))