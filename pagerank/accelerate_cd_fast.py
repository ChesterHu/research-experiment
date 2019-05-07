import numpy as np

from pagerank.accelerate_cd import AccelerateCD

class AccelerateCDFast(AccelerateCD):

    def __str__(self):
        return 'accelerated coordinate descent (fast)'

    def update_candidates(self, alpha, rho, theta, u, z, s, candidates):
        node = candidates.pop(0)
        candidates.clear()
        if self.is_candidate(node, alpha, rho, theta, u, z, s):
            candidates.append(node)
        for neighbor in self.g.neighbors(node):
            if self.is_candidate(neighbor, alpha, rho, theta, u, z, s):
                candidates.append(neighbor)