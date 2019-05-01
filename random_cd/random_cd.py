import numpy as np

from coordinate_descent import CoordinateDescent

class RandomCD(CoordinateDescent):
    def __init__(self):
        self.g = None

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        # data structures, may be not the most efficient way
        fvalues = []
        candidates = []
        q = np.zeros(self.g._num_vertices, dtype = float)
        s = np.zeros(self.g._num_vertices, dtype = float)
        gradients = np.zeros(self.g._num_vertices, dtype = float)

        # setup values, equal to step 1 of ISTA
        for node in ref_nodes:
            s[node] = 1.0 / len(ref_nodes)
            gradients[node] = -alpha * self.g.dn_sqrt[node] * s[node]
            self.update_candidates(node, rho, alpha, q, gradients, candidates)
        
        # setps from 2 to 9 of ISTA algorithm
        num_iter = 0
        threshold = (1 + epsilon) * rho * alpha

        while not self.is_terminate(gradients, threshold) and num_iter < max_iter:
            num_iter += 1
            node = self.sample(candidates)
            self.update_gradients(node, alpha, rho, q, gradients, candidates)
            fvalues.append(self.compute_fvalue(alpha, rho, q, s))

        # get approximate page rank vector
        for node in range(self.g._num_vertices):
            q[node] *= self.g.d_sqrt[node]
        
        return (q, fvalues)

    def update_gradients(self, node, alpha, rho, q, gradients, candidates):
        delta_q_node = -gradients[node] - rho * alpha * self.g.d_sqrt[node]
        q[node] += delta_q_node
        gradients[node] = -rho * alpha * self.g.d_sqrt[node] - (1 - alpha) * 0.5 * delta_q_node
        
        for neighbor in self.g.neighbors(node):
            gradients[neighbor] -= (1 - alpha) * 0.5 * self.g.dn_sqrt[neighbor] * self.g.dn_sqrt[node] * delta_q_node
            self.update_candidates(neighbor, alpha, rho, q, gradients, candidates)

    def update_candidates(self, node, alpha, rho, q, gradients, candidates):
        if node not in candidates and (q[node] - gradients[node]) >= rho * alpha * self.g.d_sqrt[node]:
            candidates.append(node)


if __name__ == "__main__":
    import os
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)
    graph_file = f'{dir_name}/data/JohnsHopkins.edgelist'
    graph_type = 'edgelist'
    separator = '\t'

    # experiment parameters
    ref_nodes = [3]
    alpha = 0.15
    rho = 1e-4
    epsilon = 1e-4
    max_iter = 1e6

    solver = RandomCD()
    solver.load_graph(graph_file, graph_type, separator)
    q, fvalues = solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

    # plot results
    import matplotlib.pyplot as plt
    plt.plot(fvalues)
    plt.xlabel('iterations')
    plt.ylabel('function values')
    plt.show()