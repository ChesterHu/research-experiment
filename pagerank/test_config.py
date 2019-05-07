import os

class TestConfig(object):

    def __init__(self, ref_nodes, alpha, rho, epsilon, max_iter, fname, ftype):
        self.ref_nodes = ref_nodes
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.graph_file = self.resolve_fname(fname)
        self.graph_type = ftype

        if not os.path.isfile(self.graph_file):
            raise IOError(f'{self.graph_file} doesn\'t exist!')

    def resolve_fname(self, fname):
        full_path = os.path.realpath(__file__)
        dir_name = os.path.dirname(full_path)
        return f'{dir_name}/analysis/data/{fname}'
