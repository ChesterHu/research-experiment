import numpy as np
import time

from .pagerank import PageRank

class RandomCD(PageRank):

    def __str__(self):
        return 'randomized coordinate descent'

    def optimize(self, alpha, rho, epsilon, max_iter, q, s, candidates, gradients, fvalues, nzeros, times):
        fvalues.append(self.compute_fvalue(alpha, rho, q, s))
        times.append(0)
        st = time.time()
        dt = 0
        num_iter = 0

        while num_iter < max_iter or times[-1] < 10:
            num_iter += 1
            node = self.sample(candidates)
            self.update_gradients(node, alpha, rho, q, gradients, candidates)

            dt += time.time() - st
            if num_iter % 1 == 0:
                times.append(dt * 1000)
                fvalues.append(self.compute_fvalue(alpha, rho, q, s))
                nzeros.append(len(np.nonzero(q)[0]))
            st = time.time()

    def update_candidates(self, node, alpha, rho, q, gradients, candidates):
        if node not in candidates and (q[node] - gradients[node]) >= rho * alpha * self.g.d_sqrt[node]:
            candidates.append(node)

    def update_gradients(self, node, alpha, rho, q, gradients, candidates):
        delta_q_node = -gradients[node] - rho * alpha * self.g.d_sqrt[node]
        q[node] += delta_q_node
        gradients[node] = -rho * alpha * self.g.d_sqrt[node] - (1 - alpha) * 0.5 * delta_q_node
        
        for neighbor in self.g.neighbors(node):
            gradients[neighbor] -= (1 - alpha) * 0.5 * self.g.dn_sqrt[neighbor] * self.g.dn_sqrt[node] * delta_q_node
            self.update_candidates(neighbor, alpha, rho, q, gradients, candidates)
