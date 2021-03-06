import numpy as np
import time
from math import sqrt

from pagerank.pagerank import PageRank

class AccelerateGD(PageRank):
    
    def __str__(self):
        return 'accelerated gradient descent'

    def optimize(self, alpha, rho, epsilon, max_iter, q, s, candidates, gradients, fvalues, nzeros, times):
        y = np.zeros(self.g._num_vertices, dtype = float)
        prev_q = np.zeros(self.g._num_vertices, dtype = float)

        fvalues.append(self.compute_fvalue(alpha, rho, q, s))
        times.append(0)
        st = time.time()
        dt = 0
        num_iter = 0

        while num_iter < max_iter or times[-1] < 10:
            num_iter += 1
            q, prev_q = prev_q, q
            beta = self.compute_beta(num_iter, alpha)
            self.update_q(alpha, rho, q, y, gradients)
            # better implementation should keep track of non-zero nodes and seed nodes.
            self.update_y(beta, q, prev_q, y)
            self.update_gradients(alpha, rho, y, s, gradients)
            
            dt += time.time() - st
            times.append(dt * 1000)
            fvalues.append(self.compute_fvalue(alpha, rho, q, s))
            nzeros.append(len(np.nonzero(y)[0]))
            st = time.time()


    def compute_beta(self, num_iter, alpha):
        if num_iter == 1:
            return 0
        else:
            return (1 - sqrt(alpha)) / (1 + sqrt(alpha))

    def update_q(self, alpha, rho, q, y, gradients):
        for node in range(self.g._num_vertices):
            z = y[node] - gradients[node]
            thd = alpha * rho * self.g.d_sqrt[node]

            if z >= thd:
                q[node] = z - thd
            elif z <= -thd:
                q[node] = z + thd
            else:
                q[node] = 0

    def update_y(self, beta, q, prev_q, y):
        for node in range(self.g._num_vertices):
            y[node] = q[node] + beta * (q[node] - prev_q[node])

    def update_gradients(self, alpha, rho, y, s, gradients):
        for node in range(self.g._num_vertices):
            gradients[node] = 0.5 * (1 + alpha) * y[node] - alpha * s[node] * self.g.dn_sqrt[node]
        
        for node in range(self.g._num_vertices):
            for neighbor in self.g.neighbors(node):
                gradients[neighbor] -= 0.5 * (1 - alpha) * y[node] * self.g.dn_sqrt[node] * self.g.dn_sqrt[neighbor]
