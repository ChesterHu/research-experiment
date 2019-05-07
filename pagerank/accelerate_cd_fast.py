import numpy as np

from pagerank.accelerate_cd import AccelerateCD

class AccelerateCDFast(AccelerateCD):

    def update_candidates(self, alpha, rho, theta, u, z, s, candidates):
        node = candidates.pop(0)
        next_node = np.random.choice(self.g.neighbors(node), 1)[0]
        candidates.append(next_node)