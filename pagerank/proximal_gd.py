import numpy as np
import time
from math import sqrt

from pagerank.accelerate_gd import AccelerateGD

class ProximalGD(AccelerateGD):
    
    def __str__(self):
        return 'gradient descent'

    def compute_beta(self, num_iter, alpha):
        # non accelerated method always has beta = 0
        return 0
