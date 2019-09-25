import numpy as np
import localgraphclustering as lgc
import matplotlib.pyplot as plt

from pagerank.proximal_gradient_descent import ProximalGradientDescent
from pagerank.accelerated_proximal_gradient_descent import AcceleratedProximalGradientDescent
from pagerank.fast_iterative_shrinkage_algorithm import FastIterativeShrinkageAlgorithm

def get_function_values(solver, iterations = 10):
    function_values = []
    for i in range(1, iterations + 1):
        solver.set_max_iter(i)
        q = solver.solve()
        function_values.append(solver.compute_function_value(q))
    return function_values

def get_number_of_non_zeros(solver, iterations = 50):
    non_zeros = []
    for i in range(1, iterations + 1):
        solver.set_max_iter(i)
        q = solver.solve()
        non_zeros.append(len(np.nonzero(q)[0]))
    return non_zeros

if __name__ == "__main__":
    graph_name = '../data/ppi_mips.graphml'
    graph_type = 'graphml'

    graph = lgc.GraphLocal(graph_name, graph_type)
    
    solver = AcceleratedProximalGradientDescent(graph, seed_nodes = [0], alpha = 0.15, epsilon = 1e-6, rho = 1e-5)
    function_values = get_function_values(solver)
    #non_zeros = get_number_of_non_zeros(solver)
    plt.plot(function_values, label = str(solver), color = 'red')
    #plt.plot(non_zeros, label = str(solver), color = 'red')


    solver = ProximalGradientDescent(graph, seed_nodes = [0], alpha = 0.15, epsilon = 1e-6, rho = 1e-5)
    function_values = get_function_values(solver)
    non_zeros = get_number_of_non_zeros(solver)
    plt.plot(function_values, label = str(solver), color = 'black')
    #plt.plot(non_zeros, label = str(solver), color = 'black')

    solver = FastIterativeShrinkageAlgorithm(graph, seed_nodes = [0], alpha = 0.15, epsilon = 1e-6, rho = 1e-5)
    function_values = get_function_values(solver)
    # non_zeros = get_number_of_non_zeros(solver)
    plt.plot(function_values, label = str(solver), color = 'green', linestyle = '--')
    #plt.plot(non_zeros, label = str(solver), color = 'green', linestyle = '--')
    
    plt.xlabel('iteration')
    plt.ylabel('number of non zero nodes')
    plt.legend()
    plt.show()