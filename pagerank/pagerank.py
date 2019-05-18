import random as rd
import numpy as np
import localgraphclustering as lgc

class PageRank(object):

    def __init__(self):
        self.g = None
    
    def load_graph(self, fname, ftype = 'edgelist', separator = '\t'):
        self.g = lgc.GraphLocal(fname, ftype, separator)

    def build_graph(self, edge_list):
        # TODO build graph from edge list
        pass

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        # data structures
        fvalues = []
        nzeros = []
        times = []
        
        candidates = []
        gradients = np.zeros(self.g._num_vertices, dtype = float)
        q = np.zeros(self.g._num_vertices, dtype = float)
        s = np.zeros(self.g._num_vertices, dtype = float)
        
        for node in ref_nodes:
            s[node] = 1.0 / len(ref_nodes)
            gradients[node] = -alpha * self.g.dn_sqrt[node] * s[node]
            candidates.append(node)

        self.optimize(alpha, rho, epsilon, max_iter, q, s, candidates, gradients, fvalues, nzeros, times)

        for node in range(self.g._num_vertices):
            q[node] = abs(q[node]) * self.g.d_sqrt[node]
            
        return (q, fvalues, nzeros, times)

    def optimize(self, alpha, rho, epsilon, max_iter, q, s, candidates, gradients, fvalues, nzeros, times):
        pass

    def update_candidates(self, node, alpha, rho, q, gradients, candidates):
        pass

    def is_terminate(self, gradients, threshold):
        max_norm = 0
        for node in range(self.g._num_vertices):
            max_norm = max(max_norm, abs(self.g.dn_sqrt[node] * gradients[node]))
        return max_norm <= threshold

    def sample(self, candidates):
        return candidates[rd.randint(0, len(candidates) - 1)]

    def compute_fvalue(self, alpha, rho, q, s):
        value = 0
        for i in range(self.g._num_vertices):
            if q[i] == 0: continue
            for j in range(self.g._num_vertices):
                if q[j] == 0: continue
                Qij = self.compute_Qij(i, j, alpha)
                value += 0.5 * q[i] * Qij * q[j]
            value += -alpha * s[i] * self.g.dn_sqrt[i] * q[i] + rho * alpha * self.g.d_sqrt[i] * abs(q[i])

        return value


    def compute_Qij(self, node_i, node_j, alpha):
        if node_i == node_j:
            Qij = (1 + alpha) * 0.5
        else:
            Qij = (-self.g.dn_sqrt[node_i] * self.g.dn_sqrt[node_j]) * (1 - alpha) * 0.5 * self.g.adjacency_matrix[node_i, node_j]
        return Qij 