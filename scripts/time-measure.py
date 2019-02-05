import numpy as np
import matplotlib.pyplot as plt
import localgraphclustering as lgc

def measureTime(methodName):
    ref_node = [3]
    for eps in np.logspace(-1, -3, 20):
        g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/JohnsHopkins.graphml", "graphml")
        print("eps: ", eps)
        lgc.approximate_PageRank(g, ref_node, epsilon = eps, method = methodName)
    print("complete method: ", methodName)

# read time file
def readTimes(fname) -> list:
    with open("../output/" + fname, 'r') as file:
        return np.array(file.read().splitlines(), dtype = np.float)

# plot execution time for both methods
def plotTime(fname):
    randTimes = readTimes("time-rand.txt")
    normTimes = readTimes("time-norm.txt")
    eps = np.logspace(-1, -3, 20)
    plt.semilogx(eps, normTimes, label = "non-random")
    plt.semilogx(eps, randTimes, label = "random")
    plt.title("running time")
    plt.xlabel("epsilon")
    plt.ylabel("time(sec)")
    plt.legend()
    plt.savefig(fname)

if __name__ == "__main__":
    # TODO delete file before generating
    measureTime('l1reg-rand')
    measureTime("l1reg")
    plotTime("../figures/run-time.png")