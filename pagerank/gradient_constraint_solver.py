import cvxpy as cp
import numpy as np

from math import sqrt
from pagerank.page_rank_solver import PageRankSolver

class GradientConstraintSolver(PageRankSolver):
    """
    Solve the problem with constraint on the gradient with or without the l1 penalty
    """
    def __str__(self):
        return 'Gradient constraint solver'
    
    def solve(self):
        """
        Solve the constraint problem with cvxpy package
        """
        q = np.zeros(self.graph._num_vertices, dtype = float)
        return self.minimize(q)

    def minimize(self, q):
        prev_q = np.copy(q)
        for _ in range(self.max_iter):
            beta = self.get_beta()
            y = q + beta * (q - prev_q)
            prev_q, q = q, prev_q
            q = self.solve_constraint_prob(y)
        return q


    def get_beta(self):
        return (1 - sqrt(self.alpha)) / (1 + sqrt(self.alpha))

    def solve_constraint_prob(self, y):
        q = cp.Variable(shape=(len(y), 1))
        gradient_y = self.compute_gradient(y)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(q - y) * 0.5 + (q - y) @ gradient_y + cp.norm(q, 1)))
        return prob.solve()