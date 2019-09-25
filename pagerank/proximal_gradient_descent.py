import numpy as np

from pagerank.page_rank_solver import PageRankSolver

class ProximalGradientDescent(PageRankSolver):
    """
    Implementation of proximal gradient descent to solve problem:
        min F(q) = f(q) + g(q)

    Assumptions:
        1. f(q) is a smooth convex function, i.e continuously differentiable with Lipschitz constant L
        2. g(q) is convex and possibly nonsmooth
        3. min F(q) is sovable
    """

    def __str__(self):
        return 'proximal gradient descent'

    def solve(self):
        """
        Minimize the objective function by proximal gradient descent
        """
        q = np.zeros(self.graph._num_vertices, dtype = float)
        return self.minimize(q, self.epsilon)

    def minimize(self, q, epsilon):
        """
        Minimize the objective function by proximal gradient descent
        """
        num_iter = 0
        while num_iter < self.max_iter:
            num_iter += 1
            gradient = self.compute_gradient(q)
            q = self.proximal_step(q, gradient)
        return q

    def proximal_step(self, q, gradient):
        """
        Perform one proximal gradient descent step and return updated q
        """
        for node in range(len(q)):
            threshold = self.rho * self.alpha * self.graph.d_sqrt[node]
            if q[node] - gradient[node] >= threshold:
                q[node] = q[node] - gradient[node] - threshold
            elif q[node] - gradient[node] <= -threshold:
                q[node] = q[node] - gradient[node] + threshold
            else:
                q[node] = 0
        return q