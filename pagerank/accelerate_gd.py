import numpy as np
import time
from math import sqrt

from pagerank.proximal_gd import ProximalGD

class AccelerateGD(ProximalGD):
    
    def __str__(self):
        return 'accelerated proximal gradient descent'

    def compute_beta(self, num_iter, alpha):
        if num_iter == 1:
            return 0
        else:
            return (1 - sqrt(alpha)) / (1 + sqrt(alpha))