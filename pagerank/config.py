import os
import yaml

from types import SimpleNamespace

class Config(object):

    def __init__(self, config_file):
        config_file = self.resolve_config_fname(config_file)
        config = yaml.safe_load(open(config_file))

        self.alpha = config.get('alpha', 0.15)
        self.epsilon = config.get('epsilon', 1e-4)
        self.rho = config.get('rho', 1e-4)
        self.max_iter = config.get('max_iter', 100)
        self.ref_nodes = config.get('ref_nodes', [0])

        self.graph_file = config.get('graph_file', '')
        self.graph_type = config.get('graph_type', '')
        if self.graph_file:
            self.graph_file = self.resolve_data_fname(self.graph_file)


    def resolve_data_fname(self, fname):
        dir_name = self.resolve_dirname(fname)
        return f'{dir_name}/analysis/data/{fname}'

    def resolve_config_fname(self, fname):
        dir_name = self.resolve_dirname(fname)
        return f'{dir_name}/analysis/{fname}'

    def resolve_dirname(self, fname):
        full_path = os.path.realpath(__file__)
        dir_name = os.path.dirname(full_path)
        return dir_name