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

# test warm start
print('test warm start')
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
nodes1, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")
# start after non-randomized method
nodes2, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg-rand")
nodes3, gradients = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")

print(f'First converge:\n{nodes1}')
print(f'Second converge:\n{nodes2}')
print(f'Third converge:\n{nodes3}')