import numpy as np
import matplotlib.pyplot as plt

import localgraphclustering as lgc

alpha = 0.15
rho = 1e-6
epsilons = np.logspace(-1, -3, 20)
ref_node = [3]

A = np.genfromtxt("../output/graph.txt", delimiter = ",")

rows, cols = A.shape
s = np.zeros((1, rows))
s[0, 3] = 1

d = A.sum(axis = 0)
D = np.diag(d)
Ds = np.diag(np.sqrt(d))
Dsinv = np.diag(1.0 / np.sqrt(d))
Q = Dsinv @ (D - (1 - alpha) / 2.0 * (D + A)) @ Dsinv

def getFuncVal(fname):
    q = np.genfromtxt(fname, delimiter = ",")
    funcVal = []
    for v in q:
        val = .5 * v @ Q @ v.T - alpha * s @ Dsinv @ v.T + rho * alpha * np.linalg.norm(Ds @ v.T, ord = 1)
        funcVal.append(val)
    return funcVal

print("compute obj function value: ")
funcVal = getFuncVal("../output/q.txt")
funcValRand = getFuncVal("../output/q-rand.txt")
plt.semilogx(epsilons, funcVal, label="non-random")
plt.semilogx(epsilons, funcValRand, label="random")
plt.ylabel("function value")
plt.xlabel("epsilon")
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig("../figures/func-value.png")