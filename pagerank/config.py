import os
import yaml

from types import SimpleNamespace

class Config(object):

    def __init__(self, config_file):
        config_file = self.resolve_conif_fname(config_file)
        config = yaml.safe_load(open(config_file))

        self.alpha = config['alpha']
        self.epsilon = config['epsilon']
        self.rho = config['rho']
        self.max_iter = config['max_iter']
        self.ref_nodes = config['ref_nodes']
        self.graph_file = self.resolve_data_fname(config['graph_file'])
        self.graph_type = config['graph_type']


    def resolve_data_fname(self, fname):
        dir_name = self.resolve_dirname(fname)
        return f'{dir_name}/analysis/data/{fname}'

    def resolve_conif_fname(self, fname):
        dir_name = self.resolve_dirname(fname)
        return f'{dir_name}/analysis/{fname}'

    def resolve_dirname(self, fname):
        full_path = os.path.realpath(__file__)
        dir_name = os.path.dirname(full_path)
        return dir_name