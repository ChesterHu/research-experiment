import numpy as np
from math import sqrt

from pagerank.page_rank_solver import PageRankSolver

class BoundedAcceleratedProximalGradientDescent(PageRankSolver):
    """
    Implementation of bounded accelerated proximal gradient descent to solve problem:
        min F(q) = f(q) + g(q)
    
    Assumption:
        1. f(q) is a smooth convex function, i.e continuously differentiable with Lipschitz constant L
        2. g(q) is convex and possibly nonsmooth
        3. min F(q) is sovable
    
    Explanation:
        Every proximal step, we force the gradient is bounded by -rho * alpha * D^(1/2)
    """

    def __str__(self):
        return 'bounded accelerated proximal gradient descent'

    def solve(self):
        """
        Solve the problem by accelerated proximal gradient descent with gradient bounded
        """
        q = np.zeros(self.graph._num_vertices, dtype = float)
        return self.minimize(q)

    def minimize(self, q):
        """
        Minimize the objective function by accelerated proximal gradient descent
        """
        prev_q = np.copy(q)
        for _ in range(self.max_iter):
            beta = self.get_beta()
            y = q + beta * (q - prev_q)
            gradient = self.compute_gradient(y)
            # when the gradient violates the bound, use non-accelerated step
            if self.is_bounded_gradient(y, gradient):
                q = self.proximal_step(q, y, gradient)
            else:
                gradient = self.compute_gradient(q)
                q = self.proximal_step(q, q, gradient)
        return q

    def is_bounded_gradient(self, q, gradient):
        """
        Return true if the gradient is bounded by -rho*alpha*D^(1/2)
        """
        for node in range(len(gradient)):
            if q[node] != 0 and gradient[node] > -self.rho * self.alpha * self.graph.d_sqrt[node]:
                return False
        return True

    def get_beta(self):
        return (1 - sqrt(self.alpha)) / (1 + sqrt(self.alpha))
    
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