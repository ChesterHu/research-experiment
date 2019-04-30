import os
import numpy as np

import localgraphclustering as lgc

class RandomCD(object):
    def __init__(self):
        self.g = None
        self.q = []
        self.gradients = []
        self.candidates = set()

    def load_graph(self, fname):
        self.g = lgc.GraphLocal(fname, 'edgelist', ' ')

    def solve(self, ref_node, alpha = 0.15, epsilon = 1e-4, rho = 1e-4):
        pass

    def update_gradients(self):
        pass

    def update_candidates(self):
        pass


if __name__ == "__main__":
    full_path = os.path.realpath(__file__)
    dir_name = os.path.dirname(full_path)

    solver = RandomCD()
    solver.load_graph(f'{dir_name}/data/Erdos02-cc.edgelist')