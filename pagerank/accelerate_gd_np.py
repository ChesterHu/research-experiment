import numpy as numpy
import scipy as scipy
import time

from pagerank.pagerank import PageRank

class AccelerateGDNumpy(object):

    def __str__(self):
        return 'accelerated gradient descent (numpy ver)'

    def solve(self, ref_nodes, alpha, rho, epsilon, max_iter):
        pass