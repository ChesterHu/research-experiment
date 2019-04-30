import os
import numpy as np

from random import randint

import localgraphclustering as lgc

class RandomCD(object):
    def __init__(self):
        self.g = None

    def load_graph(self, fname):
        self.g = lgc.GraphLocal(fname, 'edgelist', ' ')

    def solve(self, ref_node, alpha = 0.15, epsilon = 1e-4, rho = 1e-4):
        # data structures
        q = np.zeros(self.g._num_vertices, dtype = float)
        s = np.zeros(self.g._num_vertices, dtype = float)
        gradients = np.zeros(self.g._num_vertices, dtype = float)
        candidates = []

        # setup values, equal to step 1 of ISTA
        for node in ref_node:
            s[node] = 1.0 / len(ref_node)
            gradients[node] = -alpha * self.g.dn_sqrt[node] * s[node]
            self.update_candidates(node, rho, alpha, q, gradients, candidates)
        
        # setps from 2 to 9 of ISTA algorithm
        threshold = (1 + epsilon) * rho * alpha
        while not self.is_terminate(gradients, threshold):
            node = self.sample(candidates)
            self.update_gradients(node, q, gradients)
        return (q, gradients)

    def sample(self, candidates):
        return randint(0, len(candidates))

    def update_gradients(self, node, q, gradients):
        pass

    def update_candidates(self, node, rho, alpha, q, gradients, candidates):
        if node not in candidates and q[node] - gradients[node] >= rho * alpha * self.g.d_sqrt[node]:
            candidates.append(node)

    def is_terminate(self, gradients, threshold):
        max_norm = 0
        for node in range(self.g._num_vertices):
            max_norm = max(max_norm, abs(self.g.dn_sqrt[node] * gradients[node]))
        return max_norm <= threshold



if __name__ == "__main__":
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)

    solver = RandomCD()
    solver.load_graph(f'{dir_name}/data/Erdos02-cc.edgelist')