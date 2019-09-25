import numpy as np
import scipy as sp

class PageRankSolver(object):
    """
    Linear system sovler to solve variational page rank problem:
        Q = D^(-1/2) * (D - (1 - alpha) / 2 * (D + A)) * D^(1/2)
        f(q) = 1/2 * <q, Q*q> - alpha * <s, D^(-1/2) * q>
        min F(q) := rho * alpha * |D^(1/2)*q| + f(q)
    
    Explanation:
        alpha: teleportation probability
        rho: regularization parameter
        s: seed node
        D: the degree matrix of the graph
        A: adjacency matrix
        q: distribution vector
    """

    def __init__(self, graph, seed_nodes, alpha, epsilon, rho):
        """
        graph: localgraphclustering.GraphLocal object
        epsilon: termination parameter
        """
        self.graph = graph
        self.seed_nodes = seed_nodes
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho
        self.max_iter = 20

        self.s = np.zeros(self.graph._num_vertices, dtype = float)
        for node in seed_nodes:
            self.s[node] = 1.0 / len(seed_nodes)
        I = sp.sparse.diags(np.ones(self.graph._num_vertices))
        self.A = self.graph.adjacency_matrix.tocsc()
        self.D_minus_sqrt = sp.sparse.diags(self.graph.dn_sqrt)
        self.Q = ((1 + alpha) * 0.5) * I - ((1 - alpha) * 0.5) * (self.D_minus_sqrt * self.A * self.D_minus_sqrt)
        

    def compute_gradient(self, q):
        """
        Compute the gradient of f(q), formula:
            gradient_f(q) = Q * q - alpha * D^(-1/2) * s
        return the gradient of f(q)
        """
        return self.Q * q - self.alpha * (self.s * self.graph.dn_sqrt)        

    def compute_function_value(self, q):
        """
        Compute the value of f(q), formula:
            F(q) = 1/2 * <q, Q*q> - alpha * <s, D^(-1/2) * q> + rho * alpha * |D^(1/2)*q|
        return f(q)
        """
        return 0.5 * (q @ (self.Q @ q)) - self.alpha * (self.s @ (self.graph.dn_sqrt * q)) + self.alpha * self.rho * np.linalg.norm(self.graph.d_sqrt * q, 1)

    def solve(self):
        pass

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    