import numpy as np

from math import pow
from math import sqrt

from pagerank import PageRank

class AccelerateCD(PageRank):
    def __init__(self):
        self.g = None

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        # data structures
        fvalues = []
        candidates = []
        u = np.zeros(self.g._num_vertices, dtype = float)
        z = np.zeros(self.g._num_vertices, dtype = float)
        q = np.zeros(self.g._num_vertices, dtype = float)
        s = np.zeros(self.g._num_vertices, dtype = float)
        
        for node in ref_nodes:
            s[node] = 1.0 / len(ref_nodes)
            candidates.append(node)

        theta = 1.0 / self.g._num_vertices
        num_iter = 0

        while num_iter < max_iter:
            if num_iter % 1000 == 0: print(f'iter: {num_iter}...')
            num_iter += 1
            node = self.sample(candidates)
            gradient_node = self.compute_gradient(node, alpha, rho, theta, u, z, s)
            t = self.compute_t(node, gradient_node, alpha, rho, theta, z)
            self.update_uz(node, t, theta, u, z)
            self.update_candidates(theta, u, z, s, candidates)
            fvalues.append(self.compute_fvalue_accel(alpha, rho, theta, q, u, z, s))
            theta = 0.5 * (sqrt(pow(theta, 4) + 4 * pow(theta, 2)) - pow(theta, 2))

        for node in range(self.g._num_vertices):
            q[node] = (theta * theta * u[node] + z[node]) * self.g.d_sqrt[node]

        return (q, fvalues)

    def compute_gradient(self, node, alpha, rho, theta, u, z, s):
        gradient_node = 0.5 * (1 + alpha) * self.compute_y(node, theta, u, z)
        for neighbor in self.g.neighbors(node):
            gradient_node -= 0.5 * (1 - alpha) * self.compute_y(neighbor, theta, u, z) * self.g.dn_sqrt[node] * self.g.dn_sqrt[neighbor]
        gradient_node -= alpha * self.g.dn_sqrt[node] * s[node]
        return gradient_node

    def compute_fvalue_accel(self, alpha, rho, theta, q, u, z, s):
        for node in range(self.g._num_vertices):
            q[node] = self.compute_y(node, theta, u, z)

        return self.compute_fvalue(alpha, rho, q, s)

    def compute_t(self, node, gradient_node, alpha, rho, theta, z):
        lipschtz = 0.5 * (1 + alpha)
        upper = (gradient_node + rho * alpha * self.g.d_sqrt[node]) / (self.g._num_vertices * theta * lipschtz)
        lower = (gradient_node - rho * alpha * self.g.d_sqrt[node]) / (self.g._num_vertices * theta * lipschtz)

        if z[node] > upper:
            return -upper
        elif z[node] < lower:
            return -lower
        return -z[node]

    def compute_y(self, node, theta, u, z):
        return pow(theta, 2) * u[node] + z[node]

    def update_uz(self, node, t, theta, u, z):
        z[node] += t
        u[node] -= (1 - self.g._num_vertices * theta) * t / pow(theta, 2)

    def update_candidates(self, theta, u, z, s, candidates):
        node = candidates.pop(0)
        node = (node + 1) % self.g._num_vertices
        candidates.append(node)

if __name__ == "__main__":
    import os
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    graph_file = f'{dir_name}/data/JohnsHopkins.edgelist'
    graph_type = 'edgelist'
    separator = '\t'
    
    # experiment parameters
    ref_nodes = [4]
    alpha = 0.15
    rho = 1e-4
    epsilon = 1e-4
    max_iter = 10000
    
    solver = AccelerateCD()
    solver.load_graph(graph_file, graph_type, separator)
    q, fvalues = solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

    # plot results
    import matplotlib.pyplot as plt
    plt.plot(fvalues)
    plt.xlabel('iterations')
    plt.ylabel('function values')
    plt.show()