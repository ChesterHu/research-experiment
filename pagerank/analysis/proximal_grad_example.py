#!/usr/bin/env python
# coding: utf-8

# # Imports / Configuration

# In[17]:


import numpy as np

import sys,os

data_path = os.getcwd()

try:
    import localgraphclustering as lgc
except:
    # when the package is not installed, import the local version instead. 
    # the notebook must be placed in the original "notebooks/" folder
    sys.path.append("../")
    import localgraphclustering as lgc


# ## Load dataset

# In[18]:


g = lgc.GraphLocal(os.path.join(data_path,'datasets/ppi_mips.graphml'),'graphml')


# ## Define proximal accelerated gradient descent function

# In[48]:


def proximal_gradient_descent(A, dn_sqrt, d_sqrt, ref_node, rho, alpha, eps, stepsize=1.):
    
    # Number of nodes in the graph
    n = dn_sqrt.shape[0]
    
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

    # The algorithm starts here
    while ite < 1000:
        
        if ite == 0:
            beta = 0
        else:
            beta = (1-np.sqrt(alpha)/(1+np.sqrt(alpha)))
            
        # Update parameters using a gradient step
        z = y - stepsize * grad 
        
        # Store old parameters
        x_old = x
        
        # Update parameters using the proximal step        
        for i in range(n):
            if z[i] >= stepsize * rho * alpha * d_sqrt[i]:
                x[i] = z[i] - stepsize * rho * alpha * d_sqrt[i]
            elif z[i] <= -stepsize * rho * alpha * d_sqrt[i]:
                x[i] = z[i] + stepsize * rho * alpha * d_sqrt[i]
            else:
                x[i] = 0
                
        y = x + beta*(x - x_old)
        
        # Compute gradient
        grad = ((1+alpha)/2)*y  - ((1-alpha)/2)*(dn_sqrt*(A@(dn_sqrt*y))) - alpha * (seed*dn_sqrt)
    
        # Compute termination criterion
        err = np.linalg.norm(grad/d_sqrt, np.inf)
        # Increase iteration count
        ite += 1
        
    return x*d_sqrt


# ## Run proximal accelerated gradient descent on the givne graph

# In[49]:


# L1-regularization parameter for the spectral problem
rho = 1.0e-4
# Teleportation parameter of the PageRank model
alpha = 0.1
# Refence node
ref_node = 4

# Matrices and vectors that are needed by the present implementation of proximal accelerated gradient descent
A = g.adjacency_matrix
dn_sqrt = g.dn_sqrt
d_sqrt = g.d_sqrt

# Stepsize for algorithm
stepsize = 1
# Termination tolerance
epsilon = 1.0e-15

# Call proximal accelerated gradient descent
x = proximal_gradient_descent(A, dn_sqrt, d_sqrt, ref_node, rho, alpha, epsilon)


# ## Print output

# In[50]:


x


# ## Sparsity of the output

# In[51]:


np.count_nonzero(x)/g._num_vertices


# In[ ]:





# In[52]:


np.nonzero(x)[0]


# In[53]:


x[np.nonzero(x)[0]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




