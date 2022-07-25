# -*- coding: utf-8 -*-
"""
This file teaches first order numerical differentiation using finite difference methods
"""

"Auxiliary functions"
import sys 
import numpy as np 
import matplotlib.pyplot as plt

def xfunc(x) :
    """Toy function for demonstrating the finite difference methods"""
    return np.cos(x) + x*np.sin(x) 

def xfunc_dot(x):
    """Derivative of example function"""
    return x*np.cos(x)



"Display function and its derivative" 
points = 1000 # number of points
x_min = -6      # x range
x_max = -x_min  # ^^
x = np.linspace(x_min, x_max, points) # independent variable 
y = xfunc(x) # dependent variable 
y_dot = xfunc_dot(x) # derivative 

fig1 = plt.figure() 
plt.plot(x,y,'-r')
plt.plot(x,y_dot,'-b')
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.legend(['$y$','$\dot y$'],fontsize=16)

# sys.exit() # exit from the script 


"Forward finite difference"

step_size = 0.25 # step size
y_dot_forw = np.array([]) # initialize solution array
x_forw = np.array([]) # initialize step points
x0 = x_min           # initialize first point

# loop through the whole range of x
while x0 <= x_max:
    y_approx = (xfunc(x0 + step_size) - xfunc(x0))/step_size # approximate the derivative (positive)
    y_dot_forw = np.append(y_dot_forw, y_approx) # append value onto solution array
    x_forw = np.append(x_forw, x0) 
    x0 += step_size 
    
fig2 = plt.figure()
plt.plot(x,y_dot,'-r')
plt.plot(x_forw,y_dot_forw,'-b')
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot y$')
plt.legend([r'$\dot y$ true', r'$\dot y$ forward'])
# sys.exit()



"Backward finite difference"

y_dot_back = np.array([]) # initialize solution array
x_back = np.array([]) # initialize step points
x0 = x_min           # initialize first point

# loop through the whole range of x
while x0 <= x_max:
    y_approx = (xfunc(x0) - xfunc(x0 - step_size))/step_size # approximate the derivative (negative)
    y_dot_back = np.append(y_dot_back, y_approx) # append value onto solution array
    x_back = np.append(x_back, x0) 
    x0 += step_size 
    

plt.plot(x_forw,y_dot_back,'-g')
plt.legend([r'$\dot y$ true', r'$\dot y$ forward', r'$\dot y$ backward'])
# sys.exit()



"Central finite difference"

y_dot_cent = np.array([]) # initialize solution array
x_cent = np.array([]) # initialize step points
x0 = x_min           # initialize first point

# loop through the whole range of x
while x0 <= x_max:
    y_approx = (xfunc(x0 + step_size) - xfunc(x0 - step_size))/(2*step_size) # approximate the derivative (central)
    y_dot_cent = np.append(y_dot_cent, y_approx) # append value onto solution array
    x_cent = np.append(x_cent, x0) 
    x0 += step_size 
    

plt.plot(x_cent,y_dot_cent,'-k')
plt.legend([r'$\dot y$ true', r'$\dot y$ forward', r'$\dot y$ backward', r'$\dot y$ central'])
sys.exit()