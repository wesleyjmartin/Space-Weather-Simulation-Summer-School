# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:09:22 2022

@author: wesle

This file teaches first order numerical integration of ODEs using the Euler method
"""

"Auxiliary functions"
import sys 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def RHS(y, t):
    """Right hand side of the ODE dy(t)/dt = -2y(t)"""
    return -2*y

"Set up the problem" 
y0 = 3 # initial condition 
t0 = 0 # initial time 
tf = 2 # final time 

"Evaluate exact solution" 
time = np.linspace(t0,tf) # time spanned 
y_true = odeint(RHS,y0,time) # solution 

fig1 = plt.figure()
'Plot exact solution'
plt.plot(time,y_true,'k-',linewidth = 2)
plt.grid()
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend(['Truth'])
# sys.exit()

"Numerical integration" 
step_size = 0.2 # value of the fixed step size 

"First Order Runge-Kutta or Euler Method" 
current_time = t0 # initialize first timestep 
timeline = np.array([t0]) # initialize timestep points 
current_value = y0 # initialize first value
solution_rk1 = np.array([y0]) # initialize solution array

while current_time < tf-step_size :
    # Solve ODE
    next_value = current_value + step_size*RHS(current_value, current_time) 
    
    # Save Solution 
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time) 
    solution_rk1 = np.append(solution_rk1, next_value) 
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
'Plot First Order Runge-Kutta solution'
plt.plot(timeline, solution_rk1,'r-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1'])
# sys.exit()



"Second Order Runge-Kutta" 
current_time = t0 # initialize first timestep 
timeline = np.array([t0]) # initialize timestep points 
current_value = y0 # initialize first value
solution_rk2 = np.array([y0]) # initialize solution array

while current_time < tf-step_size :
    # Solve ODE
    offset_value = current_value + step_size*RHS(current_value, current_time)/2
    offset_time = current_time + step_size/2
    next_value = current_value + step_size*RHS(offset_value, offset_time) 
    
    # Save Solution 
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time) 
    solution_rk2 = np.append(solution_rk2, next_value) 
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
'Plot Second Order Runge-Kutta solution'
plt.plot(timeline, solution_rk2,'b-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2'])
# sys.exit()

"Fourth Order Runge-Kutta" 
current_time = t0 # initialize first timestep 
timeline = np.array([t0]) # initialize timestep points 
current_value = y0 # initialize first value
solution_rk4 = np.array([y0]) # initialize solution array

while current_time < tf-step_size :
    # Solve ODE
    k1 = RHS(current_value, current_time)
    k2 = RHS(current_value + k1*step_size/2, current_time + step_size/2)
    k3 = RHS(current_value + k2*step_size/2, current_time + step_size/2)
    k4 = RHS(current_value + k3*step_size, current_time + step_size)
    m = (k1 + 2*k2 + 2*k3 + k4)/6 # weighted average slope approximation
    next_value = current_value + m*step_size
    
    # Save Solution 
    next_time = current_time + step_size
    timeline = np.append(timeline, next_time) 
    solution_rk4 = np.append(solution_rk4, next_value) 
    
    # Initialize Next Step
    current_time = next_time
    current_value = next_value
    
'Plot Fourth Order Runge-Kutta solution'
plt.plot(timeline, solution_rk4,'g-o',linewidth = 2)
plt.legend(['Truth','Runge-Kutta 1','Runge-Kutta 2','Runge-Kutta 4'])
sys.exit()