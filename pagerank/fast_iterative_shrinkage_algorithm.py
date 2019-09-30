import numpy as np
from math import sqrt

from pagerank.page_rank_solver import PageRankSolver

class FastIterativeShrinkageAlgorithm(PageRankSolver):
    """
    Implementation of the fast iterative shrinkage algorithm to solve the problem:
        min F(q) = f(q) + g(q)
    
    Assumption:
        1. f(q) is a smooth convex function, i.e continuously differentiable with Lipschitz constant L
        2. g(q) is convex and possibly nonsmooth
        3. min F(q) is sovable
    """

    def __str__(self):
        return 'Standard FISTA'

    def solve(self):
        """
        Solve the problem with standard FISTA
        """
        q = np.zeros(self.graph._num_vertices, dtype = float)
        return self.minimize(q)

    def minimize(self, q):
        prev_q = np.copy(q)
        t = 1
        for _ in range(self.max_iter):
            t_next = (1 + sqrt(1 + 4 * t * t))/ 2
            y = q + (t - 1) / t_next * (q - prev_q)
            gradient = self.compute_gradient(y)
            q, prev_q = prev_q, q
            t, t_next = t_next, t
            q = self.proximal_step(q, y, gradient)
        return q

    def proximal_step(self, q, y, gradient):
        """
        Perform one proximal gradient descent step and return updated q 
        """
        for node in range(len(q)):
            threshold = self.rho * self.alpha * self.graph.d_sqrt[node]
            if y[node] - gradient[node] >= threshold:
                q[node] = y[node] - gradient[node] - threshold
            elif y[node] - gradient[node] <= -threshold:
                q[node] = y[node] - gradient[node] + threshold
            else:
                q[node] = 0
        return q
