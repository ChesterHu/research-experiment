import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

from pagerank.accelerated_proximal_gradient_descent import AcceleratedProximalGradientDescent

class BoundedAcceleratedProximalGradientDescent(AcceleratedProximalGradientDescent):
    """
    Implementation of bounded accelerated proximal gradient descent to solve problem:
        min F(q) = f(q) + g(q)
    
    Assumption:
        1. f(q) is a smooth convex function, i.e continuously differentiable with Lipschitz constant L
        2. g(q) is convex and possibly nonsmooth
        3. min F(q) is sovable
    
    Explanation:
        Every proximal step, we force the gradient is bounded by -rho * alpha * D^(1/2).
        If gradient exceeds the bound, make beta smaller
    """

    def __str__(self):
        return 'FISTA (bounded)'

    def solve(self):
        """
        Solve the problem by accelerated proximal gradient descent with gradient bounded
        """
        q = np.zeros(self.graph._num_vertices, dtype = float)
        return self.minimize(q)

    def minimize(self, q):
        """
        Minimize the objective function by accelerated proximal gradient descent
        """
        prev_q = np.copy(q)
        betas = []
        for _ in range(self.max_iter):
            beta = self.get_beta()
            while beta > 0:
                y = q + beta * (q - prev_q)
                gradient = self.compute_gradient(y)
                if self.is_bounded_gradient(q, gradient):
                    break
                beta /= 2
            betas.append(beta)
            q = self.proximal_step(q, y, gradient)
            if self.is_terminate(gradient):
                break
        self.plot_beta(betas)
        #print(f'minimum beta {min(betas)}')
        return q

    def is_bounded_gradient(self, q, gradient):
        """
        Return true if the gradient is bounded by -rho*alpha*D^(1/2)
        """
        for node in range(len(gradient)):
            if q[node] != 0 and gradient[node] > -self.rho * self.alpha * self.graph.d_sqrt[node] + self.epsilon:
                return False
        return True

    def is_terminate(self, gradient):
        return max(abs(self.graph.dn_sqrt * gradient)) <= (1 + self.epsilon) * self.rho * self.alpha

    def plot_beta(self, betas):
        const_beta = self.get_beta()
        plt.plot(betas, label='beta to bound gradient', color='red')
        plt.axhline(const_beta, label='constant beta', color='black')
        plt.xlabel('iteration')
        plt.ylabel('beta')
        plt.yscale('log')
        plt.show()