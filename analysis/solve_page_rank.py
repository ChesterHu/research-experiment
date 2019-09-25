import localgraphclustering as lgc
import matplotlib.pyplot as plt

from pagerank.proximal_gradient_descent import ProximalGradientDescent

def get_function_values(solver, iterations = 20):
    function_values = []
    for i in range(1, iterations + 1):
        solver.set_max_iter(i)
        q = solver.solve()
        function_values.append(solver.compute_function_value(q))
    return function_values

if __name__ == "__main__":
    graph_name = '../data/ppi_mips.graphml'
    graph_type = 'graphml'

    graph = lgc.GraphLocal(graph_name, graph_type)
    solver = ProximalGradientDescent(graph, [0], alpha = 0.15, epsilon = 1e-6, rho = 1e-4)
    function_values = get_function_values(solver)
    plt.plot(function_values, label = str(solver))
    plt.show()