import numpy as np

from random import randint

import localgraphclustering as lgc

class CoordinateDescent(object):
    def __init__(self):
        self.g = None
    
    def load_graph(self, fname, ftype = 'edgelist', separator = '\t'):
        self.g = lgc.GraphLocal(fname, ftype, separator)

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        pass

    def update_gradients(self, node, alpha, rho, q, gradients, candidates):
        pass
    
    def update_candidates(self, node, alpha, rho, q, gradients, candidates):
        pass

    def is_terminate(self, gradients, threshold):
        max_norm = 0
        for node in range(self.g._num_vertices):
            max_norm = max(max_norm, abs(self.g.dn_sqrt[node] * gradients[node]))
        return max_norm <= threshold

    def sample(self, candidates):
        return candidates[randint(0, len(candidates) - 1)]

    def compute_fvalue(self, alpha, rho, q, s):
        value = 0
        for i in range(self.g._num_vertices):
            if q[i] == 0: continue
            for j in self.g.neighbors(i):
                Qij = self.compute_Qij(i, j, alpha)
                value += 0.5 * q[i] * Qij * q[j]
            
            value += -alpha * s[i] * self.g.dn_sqrt[i] * q[i] + rho * alpha * self.g.d_sqrt[i] * q[i]
        return value


    def compute_Qij(self, node_i, node_j, alpha):
        Qij = -self.g.dn_sqrt[node_i] * self.g.dn_sqrt[node_j]
        if node_i == node_j:
            Qij += (1 + alpha) * 0.5
        return Qij