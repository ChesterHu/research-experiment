import time
import localgraphclustering as lgc

alpha = 0.15
eps = 1e-6
rho = 0.0001
ref_node = [x for x in range(10)]

print("loading graph...")
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
t = time.clock()
nodes, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")
print(f'non-randomized algorithm uses time: {time.clock() - t}')
print(nodes)

g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
t = time.clock()
nodes, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg-rand")
print(f'randomized algorithm uses time: {time.clock() - t}')
print(nodes)