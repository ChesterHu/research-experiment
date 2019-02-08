import localgraphclustering as lgc

alpha = 0.15
eps = 1e-3
rho = 0.00001
ref_node = [100]

print("loading graph...")
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/Colgate88_reduced.graphml", "graphml")
res = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")
print(res)

res = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg-rand")
print(res)