import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

import sys,os

data_path = os.getcwd()

try:
    import localgraphclustering as lgc
except:
    # when the package is not installed, import the local version instead. 
    # the notebook must be placed in the original "notebooks/" folder
    sys.path.append("../")
    import localgraphclustering as lgc

g = lgc.GraphLocal(os.path.join(data_path,'data/ppi_mips.graphml'),'graphml')

def proximal_gradient_descent(A, dn_sqrt, d_sqrt, Q, ref_node, rho, alpha, eps, stepsize=1.):
    
    # Number of nodes in the graph
    n = dn_sqrt.shape[0]
    
    # objective function
    f = []
    
    # Initialize seed vector
    seed = np.zeros(n)
    seed[ref_node] = 1
    
    # Initialized paramters
    x = np.zeros(n)

    # Initiliaze algorithm statistics
    err = 100.
    ite = 0
    
    y = np.copy(x)
    
    # Compute gradient
    grad = ((1+alpha)/2)*y  - ((1-alpha)/2)*(dn_sqrt*(A@(dn_sqrt*y))) - alpha * (seed*dn_sqrt)
    
    f.append((x@(Q@x))/2 -alpha*(seed@(dn_sqrt*x)) + rho*alpha*np.linalg.norm(d_sqrt*x,1))

    # The algorithm starts here
    while ite < 1000:
        
        if ite == 0:
            beta = 0
        else:
            beta = (1-np.sqrt(alpha)/(1+np.sqrt(alpha)))
            
        # Update parameters using a gradient step
        z = y - stepsize * grad 
        
        # Store old parameters
        x_old = x.copy()
        
        # Update parameters using the proximal step        
        for i in range(n):
            if z[i] >= stepsize * rho * alpha * d_sqrt[i]:
                x[i] = z[i] - stepsize * rho * alpha * d_sqrt[i]
            elif z[i] <= -stepsize * rho * alpha * d_sqrt[i]:
                x[i] = z[i] + stepsize * rho * alpha * d_sqrt[i]
            else:
                x[i] = 0
                
        f.append((x@(Q@x))/2 -alpha*(seed@(dn_sqrt*x)) + rho*alpha*np.linalg.norm(d_sqrt*x,1))
        y = x + beta*(x - x_old)
        
        # Compute gradient
        grad = ((1+alpha)/2)*y  - ((1-alpha)/2)*(dn_sqrt*(A@(dn_sqrt*y))) - alpha * (seed*dn_sqrt)
    
        # Compute termination criterion
        err = np.linalg.norm(grad/d_sqrt, np.inf)
        # Increase iteration count
        ite += 1
        
    return x*d_sqrt, f

rho = 1.0e-4
# Teleportation parameter of the PageRank model
alpha = 0.05
# Refence node
ref_node = 4

# Matrices and vectors that are needed by the present implementation of proximal gradient descent
A = g.adjacency_matrix.tocsc()
D = sp.sparse.diags(g.d,offsets=0)
Dn = sp.sparse.diags(g.dn,offsets=0)
Dn_sqrt = sp.sparse.diags(g.dn_sqrt,offsets=0)
I = sp.sparse.diags(np.ones(g._num_vertices),offsets=0)
Q = ((1+alpha)/2)*I - ((1-alpha)/2)*(Dn_sqrt*A*Dn_sqrt)
d = g.d
dn_sqrt = g.dn_sqrt
d_sqrt = g.d_sqrt

# Stepsize for algorithm
stepsize = 2/(1+alpha)
# Termination tolerance
epsilon = 1.0e-4

start2 = time.time()
x, f = proximal_gradient_descent(A, dn_sqrt, d_sqrt, Q, ref_node, rho, alpha, epsilon)
end2 = time.time()
print(end2 - start2)

q = x * dn_sqrt
t = time.time()
val1 = (q@(Q@q))/2
t = time.time() - t
print(f'numpy compute qQq time: {t}, value: {val1}')

t = time.time()
val2 = 0
for i in range(len(q)):
    if q[i] == 0: continue
    for j in range(len(q)):
        if q[j] == 0: continue
        val2 += q[i] * q[j] * Q[i, j] * 0.5
t = time.time() - t
print(f'loop compute qQq time: {t}, value: {val2}')
