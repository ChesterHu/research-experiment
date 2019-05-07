"""
This script produce plot for comparing function values in both iteration.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_cd_fast import AccelerateCDFast

full_path = os.path.realpath(__file__)
dir_name = os.path.dirname(full_path)
graph_file = f'{dir_name}/data/ppi_mips.graphml'
graph_type = 'graphml'

# experiment parameters
ref_nodes = [4]
alpha = 0.1
rho = 1e-4
epsilon = 1e-8
max_iter = 1000

def plot_fvalues(solver, config):
    pass

# solve accelerated method
accel_solver = AccelerateCD()
accel_solver.load_graph(graph_file, graph_type)
accel_q, accel_fvalues, accel_nzeros = accel_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)
# solve non-accelerated method
rand_solver = RandomCD()
rand_solver.load_graph(graph_file, graph_type)
rand_q, rand_fvalues, rand_nzeros = rand_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)
# solve accelerated method fast version
fast_solver = AccelerateCDFast()
fast_solver.load_graph(graph_file, graph_type)
fast_q, fast_fvalues, fast_nzeros = fast_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

# plot function value in both methods
plt.plot([_ + 1 for _ in range(len(accel_fvalues))], accel_fvalues, label = 'accelerated method', linestyle = 'solid', linewidth = 3, color = 'red')
plt.plot([_ + 1 for _ in range(len(rand_fvalues))], rand_fvalues, label = 'non-accelerated method', linestyle = 'dashed', linewidth = 3, color = 'black')
plt.plot([_ + 1 for _ in range(len(fast_fvalues))], fast_fvalues, label = 'fast accelerated method', linestyle = 'dotted', linewidth = 3, color = 'green')

# plot settings
plt.xscale('log')
plt.legend(prop={'size': 18})
plt.xlabel('iterations', fontsize = 20)
plt.ylabel('function value', fontsize = 20)
plt.show()