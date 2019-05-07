import matplotlib.pyplot as plt
import numpy as np
import os

from pagerank.accelerate_cd import AccelerateCD
from pagerank.accelerate_cd_fast import AccelerateCDFast
from pagerank.accelerate_gd import AccelerateGD

full_path = os.path.realpath(__file__)
dir_name = os.path.dirname(full_path)
graph_file = f'{dir_name}/data/ppi_mips.graphml'
graph_type = 'graphml'
separator = ''

# experiment parameters
ref_nodes = [4]
alpha = 0.05
rho = 1e-4
epsilon = 1e-4
max_iter = 1000

gd_solver = AccelerateGD()
gd_solver.load_graph(graph_file, graph_type)
gd_q, gd_fvalues, gd_nzeros = gd_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

cd_solver = AccelerateCDFast()
cd_solver.load_graph(graph_file, graph_type)
cd_q, cd_fvalues, cd_nzeros = cd_solver.solve(ref_nodes, alpha, rho, epsilon, max_iter)

iterations = [_ for _ in range(1, max_iter + 2)]
plt.plot(iterations, gd_fvalues, label = 'gradient descent', linestyle = 'dashed', linewidth = 3, color = 'black')
plt.plot(iterations, cd_fvalues, label = 'coordinate descent', linestyle = 'solid', linewidth = 3, color = 'red')
plt.legend(prop = {'size': 18})

plt.xlabel('iterations', fontsize = 20)
plt.ylabel('function value', fontsize = 20)
plt.xscale('log')
plt.show()