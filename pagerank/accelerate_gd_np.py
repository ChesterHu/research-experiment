import numpy as np
import scipy as sp
import time

from pagerank.accelerate_gd import AccelerateGD

class AccelerateGDNumpy(AccelerateGD):

    def __str__(self):
        return 'accelerated gradient descent (numpy ver)'

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        fvalues = []
        nzeros = []
        times = []

        A = self.g.adjacency_matrix.tocsc()
        I = sp.sparse.diags(np.ones(self.g._num_vertices))
        self.g.dn_sqrt = sp.sparse.diags(self.g.dn_sqrt)
        Q = ((1 + alpha) * 0.5) * I - ((1 - alpha) * 0.5) * (self.g.dn_sqrt * A * self.g.dn_sqrt)

        q = np.zeros(self.g._num_vertices, dtype = float)
        s = np.zeros(self.g._num_vertices, dtype = float)
        y = np.zeros(self.g._num_vertices, dtype = float)
        prev_q = np.zeros(self.g._num_vertices, dtype = float)

        for node in ref_nodes:
            s[node] = 1.0 / len(ref_nodes)

        num_iter = 0
        fvalues.append(self.compute_fvalue(Q, alpha, rho, q, s))
        times.append(0)
        st = time.time()
        dt = 0
        while num_iter < max_iter: # or times[-1] < 10:
            num_iter += 1
            q, prev_q = prev_q, q
            beta = self.compute_beta(num_iter, alpha)
            gradient = self.compute_gradient(A, alpha, y, s)
            self.update_q(alpha, rho, q, y, gradient)
            y = q + beta * (q - prev_q)

            dt += time.time() - st
            times.append(dt * 1000)
            fvalues.append(self.compute_fvalue(Q, alpha, rho, q, s))
            nzeros.append(len(np.nonzero(y)[0]))
            st = time.time()
        
        q *= self.g.d_sqrt
        return (q, fvalues, nzeros, times)

    def compute_gradient(self, A, alpha, y, s):
        gradient = ((1 + alpha) / 2) * y  
        gradient -= ((1 - alpha) / 2) * (self.g.dn_sqrt * (A @ (self.g.dn_sqrt * y))) 
        gradient -= alpha * (s * self.g.dn_sqrt)
        return gradient


    def compute_fvalue(self, Q, alpha, rho, q, s):
        fvalue = 0.5 * (q @ (Q @ q))
        fvalue += -alpha * (s @ (self.g.dn_sqrt * q)) 
        fvalue += alpha * rho * np.linalg.norm(self.g.d_sqrt * q, 1)
        return fvalue