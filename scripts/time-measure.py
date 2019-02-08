import os
import time
import numpy as np
import matplotlib.pyplot as plt
import localgraphclustering as lgc

outputPath = "../output/"
randFile = "time-rand.txt"
normFile = "time-norm.txt"
randqFile = "q-rand.txt"
normqFile = "q-norm.txt"
DatasetName = "JohnsHopkins"

ref_node = [3]
alphas = np.linspace(0.1, 0.5, 50)
rhos = np.logspace(-4, -7, 50)
epsilons = np.logspace(-2, -6, 50)

print("loading graph...")
g = lgc.GraphLocal("../../LocalGraphClustering/notebooks/datasets/{0}.edgelist".format(DatasetName))

def measureTimeEpsilon(methodName):
    for eps in epsilons:
        print("eps: ", eps)
        lgc.approximate_PageRank(g, ref_node, epsilon = eps, method = methodName)

def measureTimeAlpha(methodName):
    for alpha in alphas:
        print("alpha: ", alpha)
        lgc.approximate_PageRank(g, ref_node, alpha = alpha, method = methodName)

def measureTimeRho(methodName):
    for rho in rhos:
        print("rho: ", rho)
        lgc.approximate_PageRank(g, ref_node, rho = rho, method = methodName)

# read time file
def readTimes(fname):
    with open(fname, 'r') as file:
        return np.array(file.read().splitlines(), dtype = np.float)

# clean up file
def cleanUp(fnames):
    for fname in fnames:
        if os.path.isfile(fname):
            os.remove(fname)

# plot execution time for both methods
def plotTime(fname, measureTime, x, label):
    # clean existing files
    outputRandFile = outputPath + randFile
    outputNormFile = outputPath + normFile
    outputqRandFile =outputPath + randqFile
    outputqNormFile = outputPath + normqFile
    cleanUp([outputRandFile, outputNormFile, outputqRandFile, outputqNormFile])
    # run algorithms
    measureTime("l1reg-rand")
    measureTime("l1reg")
    # read data and plot execution time
    randTimes = readTimes(outputRandFile)
    normTimes = readTimes(outputNormFile)
    plt.semilogx(x, normTimes, label = "non-random")
    plt.semilogx(x, randTimes, label = "random")
    plt.title("running time")
    plt.xlabel(label)
    plt.ylabel("time(sec)")
    plt.legend()
    plt.savefig(fname)
    plt.close()

if __name__ == "__main__":
    plotTime("../figures/run-time-eps-{}-seed{}.png".format(DatasetName, ref_node[0]), measureTimeEpsilon, epsilons, "epsilon")
    plotTime("../figures/run-time-alpha-{}-seed{}.png".format(DatasetName, ref_node[0]), measureTimeAlpha, alphas, "alpha")
    plotTime("../figures/run-time-rho-{}-seed{}.png".format(DatasetName, ref_node[0]), measureTimeRho, rhos, "rho")

