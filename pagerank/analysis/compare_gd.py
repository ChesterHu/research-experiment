"""
Test non-zero nodes in gradient descent and accelerated gradient descent
"""
import matplotlib.pyplot as plt

from nzeros import plot_nzeros
from fvalues import plot_fvalues

from pagerank.accelerate_gd import AccelerateGD
from pagerank.proximal_gd import ProximalGD
from pagerank.config import Config

if __name__ == "__main__":
    
    # load configs
    config_file = 'config.yaml'
    config = Config(config_file)

    # plot
    fontsize = 20
    legendsize = 20

    # plot non-zero nodes
    plt.subplot(1, 2, 1)
    plot_nzeros(AccelerateGD(), linestyle = 'solid', color = 'red', config = config)
    plot_nzeros(ProximalGD(), linestyle = 'dashed', color = 'black', config = config)
    plt.legend(prop = {'size': legendsize})

    # plot function values
    plt.subplot(1, 2, 2)
    plot_fvalues(AccelerateGD(), linestyle = 'solid', color = 'red', config = config)
    plot_fvalues(ProximalGD(), linestyle = 'dashed', color = 'black', config = config)
    plt.legend(prop = {'size': legendsize})


    plt.show()