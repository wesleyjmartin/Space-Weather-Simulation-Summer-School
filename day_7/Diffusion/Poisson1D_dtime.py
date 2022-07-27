# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:01:09 2022

@author: wesle
"""

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



Narr = [16] # different values of N
errorarr = []

for i in Narr: # loop over several numbers of points (N)
    "Number of points"
    N = i
    Dx = 1/N
    x = np.linspace(0,1,N+1)
    xN = np.linspace(0,1+Dx,N+2) # domain with ghost state 
    
    "Time parameters"
    dt = 1/24 
    time = np.arange(0,3+dt,dt)
    nt = np.size(time)

    "Order of Neumann boundary condition approximation"
    order = 2
    
    "Initialize solution U"
    if order<2:
        U = np.zeros((N+1,nt))
        # U[:,0] = x*(3-2*x)*np.exp(x)+u0
    else:
        
        U = np.zeros((N+2,nt))

    "Solution at each timestep"
    for it in range(0,nt-1):  
        if order<2:   # first-order approximation
            "System matrix and RHS term"
            A = (1/Dx**2)*(2*np.diag(np.ones(N+1)) - np.diag(np.ones(N),-1) - np.diag(np.ones(N),1))
            F = 2*(2*x**2 + 5*x - 2)*np.exp(x)
            
            "Temporal term"
            A += (1/dt)*np.diag(np.ones(N+1))
            F += U[:,it]/dt
            
            "Boundary condition at x=0"
            A[0,:] = np.concatenate(([1],np.zeros(N)))
            F[0] = u0
            
            # Boundary condition at x=1    
            A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-1),[-1,1]))
            F[N] = 0
            
        else: # second-order approximation 
            "System matrix and RHS term"
            A = (1/Dx**2)*(2*np.diag(np.ones(N+2)) - np.diag(np.ones(N+1),-1) - np.diag(np.ones(N+1),1))
            F = 2*(2*xN**2 + 5*xN - 2)*np.exp(xN)
            
            "Temporal term"
            A += (1/dt)*np.diag(np.ones(N+2))
            F += U[:,it]/dt
            
            "Boundary condition at x=0"
            A[0,:] = np.concatenate(([1],np.zeros(N+1)))
            F[0] = 0
            
            "Boundary condition at x=1"   
            # A[N,:] = (1/Dx)*np.concatenate((np.zeros(N-2),[1/2,-2,3/2]))
            A[N+1,:] = (0.5/Dx)*np.concatenate((np.zeros(N-1),[-1, 0, 1])) # centered differences approximation
            F[N+1] = 0
        
        "Solution of the linear system AU=F"
        u = np.linalg.solve(A, F)
        U[:,it+1] = u
        # u = np.concatenate(([0],U,[0]))
        
    u = u[0:N+1]
    ua = 2*x*(3-2*x)*np.exp(x)

    # "Plotting solution"
    # plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
    # plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
    # plt.legend(fontsize=12,loc='upper left')
    # plt.grid()
    # # plt.axis([0, 1.05, 0, 6])
    # plt.xlabel("x",fontsize=16)
    # plt.ylabel("u",fontsize=16)
    
    "Animation of the results"
    fig = plt.figure()
    ax = plt.axes(xlim = (0,1), ylim = (u0, u0+6))
    myAnimation, =
    
    "Compute error"
    error = np.max(np.abs(u-ua))
    print("Linf error u: %g\n" % error)

    errorarr.append(error) # save the error for this number of points
print(errorarr[1]/errorarr[0], errorarr[2]/errorarr[1]) # print the ratios of the errors


