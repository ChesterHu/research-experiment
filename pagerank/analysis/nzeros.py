"""
This script produce plot for comparing number of non-zero nodes in both methods. 
The accelerated method can touch more non-zero nodes than non-accelerated method, but it uses less iterations to converge.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from pagerank.random_cd import RandomCD
from pagerank.accelerate_cd import AccelerateCD

full_path = os.path.realpath(__file__)
dir_name = os.path.dirname(full_path)
graph_file = f'{dir_name}/data/ppi_mips.graphml'
graph_type = 'graphml'
separator = ''

# experiment parameters
ref_nodes = [4]
alpha = 0.1
rho = 1e-4
epsilon = 1e-8
max_iter = 1000

# solve accelerated method
accel_solver = AccelerateCD()
accel_solver.load_graph(graph_file, graph_type)
accel_q, accel_fvalues, accel_nzeros = accel_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)
# solve non-accelerated method
rand_solver = RandomCD()
rand_solver.load_graph(graph_file, graph_type, separator)
rand_q, rand_fvalues, rand_nzeros = rand_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

# plot number of non-zero nodes in both methods and horizontal line for number of non-zero nodes in optimal solution
plt.plot([_ + 1 for _ in range(len(accel_nzeros))], accel_nzeros, label = 'accelerated method', linestyle = 'solid', linewidth = 3, color = 'red')
plt.plot([_ + 1 for _ in range(len(rand_nzeros))], rand_nzeros, label = 'non-accelerated method', linestyle = 'dashed', linewidth = 3, color = 'black')
plt.axhline(y = len(np.nonzero(rand_q)[0]), label = 'optimal solution', linestyle = 'dotted', linewidth = 3, color = 'blue')

# plot settings
plt.xscale('log')
plt.legend(prop={'size': 18})
plt.xlabel('iterations', fontsize = 20)
plt.ylabel('number of non-zero nodes', fontsize = 20)
plt.show()