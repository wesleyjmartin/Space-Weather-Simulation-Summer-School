# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:06:51 2022

@author: wesle

Plotting a nonlinear pendulum with state space 
(rewriting 2nd order differential as 1st order differential with state space)
"""

"Auxiliary functions"
import sys 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint 


def pendulum_free(x, t):
    """
    Dynamics of a nonlinear pendulum without any constraint. 
    Parameters
    ----------
    x : state space variable, 2-element vector
    t : time
    
    Returns
    -------
    x_dot : velocity
    """
    g = 9.81 # gravity (m/s)
    l = 3 # length of pendulum (m)
    xdot = np.zeros(2)# initialize vector of 2
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0])
    return xdot

def pendulum_damped(x, t):
    """
    Dynamics of a nonlinear pendulum with a damper. 
    Parameters
    ----------
    x : state space variable, 2-element vector 
    t : time
    
    Returns
    -------
    x_dot : velocity
    """
    g = 9.81 # gravity (m/s)
    l = 3 # length of pendulum (m)
    damp = 0.3 # damper coefficient 
    xdot = np.zeros(2) # initialize vector of 2
    xdot[0] = x[1]
    xdot[1] = -g/l*np.sin(x[0]) - damp*x[1]
    return xdot

def RK2(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 2"""
    n = len(t)
    y = np.zeros((n,len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i] # timestep
        k1 = func(y[i], t[i])
        k2 = func(y[i] + k1*h/2., t[i] + h/2)
        y[i+1] = y[i] + k2*h
    return y

def RK4(func, y0, t):
    """Explicit Integrator Runge-Kutta Order 4"""
    n = len(t)
    y = np.zeros((n,len(y0)))
    y[0] = y0
    for i in range(n-1):
        h = t[i+1] - t[i] # timestep 
        k1 = func(y[i], t[i])
        k2 = func(y[i] + k1*h/2., t[i] + h/2)
        k3 = func(y[i] + k2*h/2., t[i] + h/2)
        k4 = func(y[i] + k3*h, t[i] + h)
        m = (k1 + 2*k2 + 2*k3 + k4)/6
        y[i+1] = y[i] + m*h
    return y


"Propagate Free Pendulum" 
x0 = np.array([np.pi/3,0]) # initial conditions
t0 = 0.0 
tf = 15.0 
n_points = 1000 
time = np.linspace(t0, tf, n_points)
y = odeint(pendulum_free, x0, time)

"Display"
fig1 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y[:,0],'k-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')

plt.subplot(2,1,2)
plt.plot(time,y[:,1],'k-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
# sys.exit()



"Propagate Damped Pendulum" 
x0 = np.array([np.pi/3,0]) # initial conditions
n_points = 25
time_new = np.linspace(t0, tf, n_points)
y = odeint(pendulum_damped,x0,time)
y_rk2 = RK2(pendulum_damped,x0,time_new)
y_rk4 = RK4(pendulum_damped,x0,time_new)

"Display"
fig2 = plt.figure()
plt.subplot(2,1,1)
plt.plot(time,y[:,0],'k-',linewidth = 2)
plt.plot(time_new,y_rk2[:,0],'b-',linewidth = 2)
plt.plot(time_new,y_rk4[:,0],'g-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')

plt.subplot(2,1,2)
plt.plot(time,y[:,1],'k-',linewidth = 2)
plt.plot(time_new,y_rk2[:,1],'b-',linewidth = 2)
plt.plot(time_new,y_rk4[:,1],'g-',linewidth = 2)
plt.grid()
plt.xlabel('time [s]')
plt.ylabel(r'$\theta$')
sys.exit()