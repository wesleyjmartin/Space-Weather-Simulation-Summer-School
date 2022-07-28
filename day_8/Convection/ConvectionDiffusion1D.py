#!/usr/bin/env python
"""
Solution of a 1D Convection-Diffusion equation: -nu*u_xx + c*u_x = f
Domain: [0,1]
BC: u(0) = u(1) = 0
with f = 1

Analytical solution: (1/c)*(x-((1-exp(c*x/nu))/(1-exp(c/nu))))

Finite differences (FD) discretization:
    - Second-order cntered differences advection scheme

"""
__author__ = 'Jordi Vila-Pérez'
__email__ = 'jvilap@mit.edu'


import numpy as np
import matplotlib.pyplot as plt
from math import pi
%matplotlib qt
plt.close()
import matplotlib.animation as animation

"Flow parameters"
nu = 0.01
c = 2

# Narr = [128,256,512]
# errorarr = []

# for i in Narr:

"Number of points"
N = 32
Dx = 1/N
x = np.linspace(0,1,N+1)


"Solver parameters"
order = 1

"Temporal parameters"
dt = 0.1
time = np.arange(0,3,dt)
nt = np.size(time)
    
"Initialize U"
U = np.zeros((N-1,nt))

"Solution at each time step"
for it in range(0,nt-1):
    
    "System matrix and RHS term"
    "Diffusion term"
    Diff = nu*(1/Dx**2)*(2*np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1) - np.diag(np.ones(N-2),1))  
    "Advection term"
    if order < 2:
        cp = max(c,0)
        cm = min(c,0)
        
        "Advection term: first-order upwind"
        Advp = cp*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),-1))
        Advm = cm*(np.diag(np.ones(N-1)) - np.diag(np.ones(N-2),1))
    else:    
        "Advection term: centered differences"
        Advp = -0.5*c*np.diag(np.ones(N-2),-1)
        Advm = -0.5*c*np.diag(np.ones(N-2),1)
    
    Adv = (1/Dx)*(Advp-Advm)
    A = Diff + Adv
    "Source term"
    F = np.ones(N-1)
    
    "Temporal terms"
    U0 = U[:,it]
    A += (1/dt)*np.diag(np.ones(N-1))
    F += U0/dt
    
    "Solution of the linear system AU=F"
    u = np.linalg.solve(A,F)
    U[:,it+1] = u
    u = np.concatenate(([0],U[0:N+1,it],[0]))

"Solution of the linear system AU=F"
# U = np.linalg.solve(A,F)
# u = np.concatenate(([0],U,[0]))
ua = (1/c)*(x-((1-np.exp(c*x/nu))/(1-np.exp(c/nu))))

"Animation of the results"
fig = plt.figure()
ax = plt.axes(xlim =(0, 1),ylim =(0,1/c)) 
plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
myAnimation, = ax.plot([], [],':ob',linewidth=2)
plt.grid()
plt.xlabel("x",fontsize=16)
plt.ylabel("u",fontsize=16)

def animate(i):
    
    u = np.concatenate(([0],U[0:N+1,i],[0]))
    plt.plot(x,u)
    myAnimation.set_data(x, u)
    return myAnimation,

anim = animation.FuncAnimation(fig,animate,frames=range(1,nt),blit=True,repeat=False)

# plt.plot(x,ua,'-r',linewidth=2,label='$u_a$')
# plt.plot(x,u,':ob',linewidth=2,label='$\widehat{u}$')
# plt.legend(fontsize=12,loc='upper left')
# plt.grid()
# plt.axis([0, 1,0, 2/c])
# plt.xlabel("x",fontsize=16)
# plt.ylabel("u",fontsize=16)


"Compute error"
error = np.max(np.abs(u-ua))
print("Linf error u: %g\n" % error)

"Peclet number"
P = np.abs(c*Dx/nu)
print("Pe number Pe=%g\n" % P);

#     errorarr.append(error)

# print(errorarr[1]/errorarr[0], errorarr[2]/errorarr[1])
