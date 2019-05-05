import numpy as np

from math import sqrt

from .pagerank import PageRank

class AcceleratedGD(PageRank):
    
    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        # data structures
        fvalues = []
        nzeros = []
        nzero_nodes = set()
        q = np.zeros(self.g._num_vertices, dtype = float)
        s = np.zeros(self.g._num_vertices, dtype = float)
        y = np.zeros(self.g._num_vertices, dtype = float)
        prev_q = np.zeros(self.g._num_vertices, dtype = float)
        gradients = q = np.zeros(self.g._num_vertices, dtype = float)

        for node in ref_nodes:
            s[node] = 1.0 / len(ref_nodes)
            gradients[node] = -alpha * self.g.dn_sqrt[node] * s[node]
            nzero_nodes.add(node)

        num_iter = 0
        while num_iter < max_iter:
            num_iter += 1
            q, prev_q = prev_q, q
            beta = self.compute_beta(num_iter, alpha)
            self.update_q(alpha, rho, q, prev_q, y, gradients)
            self.update_y(beta, q, prev_q, y, nzero_nodes)
            self.update_gradients(alpha, rho, y, s, gradients, nzero_nodes)
            # record status
            fvalues.append(self.compute_fvalue(alpha, rho, q, s))
            nzeros.append(len(nzeros))

        for node in range(len(self.g._num_vertices)):
            q[node] = q[node] * self.g.d_sqrt[node]
        
        return (q, fvalues, nzeros)

    def compute_beta(self, num_iter, alpha):
        if num_iter == 1:
            return 1
        else:
            return (1 - sqrt(alpha)) / (1 + sqrt(alpha))

    def update_q(self, alpha, rho, q, prev_q, y, gradients):
        for node in range(len(self.g._num_vertices)):
            z = y[node] - gradients[node]
            thd = alpha * rho * self.g.d_sqrt[node]

            if z >= thd:
                q[node] = z - thd
            elif z <= -thd:
                q[node] = z + thd
            else:
                q[node] = 0

    def update_y(self, beta, q, prev_q, y, nzero_nodes):
        for node in range(len(self.g._num_vertices)):
            y[node] = q[node] + beta * (q[node] - prev_q[node])
            if y[node] != 0:
                nzero_nodes.add(node)

    def update_gradients(self, alpha, rho, y, s, gradients, nzero_nodes):
        for node in nzero_nodes:
            gradients[node] = 0.5 * (1 + alpha) * y[node] - alpha * s[node] * self.g.dn_sqrt[node]
        
        for node in nzero_nodes:
            for neighbor in self.g.neighbors(node):
                gradients[neighbor] -= 0.5 * (1 - alpha) * y[node] * self.g.dn_sqrt[node] * self.g.dn_sqrt[neighbor]