#!/usr/bin/env python
"""
Solution of a 1D Poisson equation: -u_xx = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = (3*x + x^2)*exp(x)

Analytical solution: -x*(x-1)*exp(x)

Finite differences (FD) discretization: second-order diffusion operator
"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    # We have to load this
from math import pi
%matplotlib qt
plt.close()

# Specify order of approximation of BC
order = 2
Narr = [8,16,32] # different values of N
errorarr = []

for i in Narr: # loop over several numbers of points (N)
    "Number of points"
    N = i
    Dx = 1/N
    x = np.linspace(0,1,N+1)
    xN = np.linspace(0,1+Dx,N+2) # domain with ghost state 
    
    if order == 1:   # first-order approximation
        "System matrix and RHS term"
        A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
        F = 2*(2*x**2 + 5*x - 2)*np.exp(x)
        
        # Boundary condition at x=0
        A[0,:] = np.concatenate(([1],np.zeros(N)))
        F[0] = 0
        
        # Boundary condition at x=1    
        A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
        F[N] = 0
        
    elif order == 2: # second-order approximation 
        "System matrix and RHS term"
        A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
        F = 2*(2*xN**2 + 5*xN - 2)*np.exp(xN)
        
        # Boundary condition at x=0
        A[0,:] = np.concatenate(([1],np.zeros(N+1)))
        F[0] = 0
        
        # Boundary condition at x=1    
        # A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2,3/2]))
        A[N+1,:] = (0.5/Dx)*np.concatenate((np.zeros(N-1),[-1, 0, 1])) # centered differences approximation
        F[N+1] = 0
    
    "Solution of the linear system AU=F"
    U = np.linalg.solve(A,F)
    # u = np.concatenate(([0],U,[0]))
    u = U
    ua = 2*x*(3-2*x)*np.exp(x)

    "Plotting solution"
    plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
    plt.plot(x,u[:-1],':ob',linewidth=2,label='$\widehat{u}$')
    plt.legend(fontsize=12,loc='upper left')
    plt.grid()
    plt.axis([0, 1.05, 0, 6])
    plt.xlabel("x",fontsize=16)
    plt.ylabel("u",fontsize=16)
    
    "Compute error"
    error = np.max(np.abs(u[:-1]-ua))
    print("Linf error u: %g\n" % error)

    errorarr.append(error) # save the error for this number of points
print(errorarr[1]/errorarr[0], errorarr[2]/errorarr[1]) # print the ratios of the errors

#%%
# For future reference: this is how you make a new cell

