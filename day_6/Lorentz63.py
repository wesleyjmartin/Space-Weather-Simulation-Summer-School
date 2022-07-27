# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:04:51 2022

@author: wesle
"""

"Auxiliary functions"
import sys 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

"""Assignment 1"""

def Lorentz63(x,t,sigma,rho,beta):
    """
    The Lorentz63 system for different coefficients. 

    Parameters
    ----------
    x : vector
        Vector of the form [x,y,z].
    t : vector
        Time vector.
    sigma : constant
        System parameter.
    rho : constant
        System parameter.
    beta : constant
        System parameter.

    Returns
    -------
    x_dot : vector
        Velocity vector of the form [x,y,z].

    """
    # x-axis = x[0], y-axis = x[1], z-axis = x[2]
    x_dot = np.zeros(3)
    x_dot[0] = sigma*(x[1] - x[0]) # x dot
    x_dot[1] = x[0]*(rho - x[2]) - x[1] # y dot
    x_dot[2] = x[0]*x[1] - beta*x[2] # z dot
    return x_dot 


"Set parameters"
sigma = 10 
rho = 28 
beta = 8/3

x0 = 5*np.ones(3) # initial state
t_in = 0 # initial time
t_fin = 20# final time
n_points = 1000 
time = np.linspace(t_in, t_fin, n_points) # time vector

# Solve ODE 
y = odeint(Lorentz63,x0,time,args = (sigma,rho,beta))

"Display" 
fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(y[:,0], y[:,1], y[:,2], 'b') # plot 3d solution
# sys.exit()



"""Assignment 2"""

"Randomly select 20 initial conditions" 
x_init = np.random.uniform(-20,20,20) # range = [-20,20]
y_init = np.random.uniform(-30,30,20) # range = [-30,30]
z_init = np.random.uniform(0,50,20) # range = [0,50]

fig2 = plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(x_init)-1):
    x0 = np.array([x_init[i],y_init[i],z_init[i]])
    # Solve ODE 
    y = odeint(Lorentz63,x0,time,args = (sigma,rho,beta))
    ax.plot3D(y[:,0], y[:,1], y[:,2]) # plot 3d solution
sys.exit()