import localgraphclustering as lgc

alpha = 0.15
eps = 1e-6
rho = 0.00001
ref_node = [3]

print("loading graph")
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/JohnsHopkins.graphml", "graphml")
print("random algorithm")
res = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg-rand")
print(res)

print("loading graph")
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/JohnsHopkins.graphml", "graphml")
print("norm algorithm")
res = lgc.approximate_PageRank(g, ref_node, alpha = alpha, rho = rho, epsilon = eps, method = "l1reg")
print(res)
