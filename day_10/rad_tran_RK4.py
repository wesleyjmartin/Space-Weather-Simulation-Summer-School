# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:56:09 2022

@author: wesle
"""

#!/usr/bin/env python

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from helper_functions import *
from physics_functions_RK4 import *

#-----------------------------------------------------------------------------
# some constants we may need
#-----------------------------------------------------------------------------

mass_o = 16.0 # in AMU
mass_o2 = 32.0 # in AMU
mass_n2 = 28.0 # in AMU

#-----------------------------------------------------------------------------
# Initialize temperature as a function of altitude
#-----------------------------------------------------------------------------

def init_temp(alt_in_km):
    temp_in_k = 200 + 600 * np.tanh( (alt_in_km - 100) / 100.0)
    return temp_in_k

#-----------------------------------------------------------------------------
# Update the old state (temp, densities), dTdt, and dt:
#-----------------------------------------------------------------------------

def update_state(old_state, dTdt, dt):

    new_state = {}

    return new_state

# ----------------------------------------------------------------------
# Main Code
# ----------------------------------------------------------------------

if __name__ == '__main__':  # main code block

    # set f107:
    f107 = 100.0
    f107a = 100.0

    SZA = 0.0
    efficiency = 0.3
    
    # boundary conditions for densities:
    n_o_bc = 5.0e17 # /m3
    n_o2_bc = 4.0e18 # /m3
    n_n2_bc = 1.7e19 # /m3

    # read in EUV file (step 1):
    euv_file = 'euv_37.csv'
    euv_info = read_euv_csv_file(euv_file)
    # print('short : ', euv_info['short'])
    # print('ocross : ', euv_info['ocross'])

    # initialize altitudes (step 2):
    nAlts = 41
    alts = np.linspace(100, 500, num = nAlts)
    # print('alts : ', alts)


    def RHS(temp, time):
        # compute scale height in km (step 4):
        h_o = calc_scale_height(mass_o, alts, temp)
        # print('Scale height of o : ', h_o)
        h_o2 = calc_scale_height(mass_o2, alts, temp)
        # print('Scale height of o2 : ', h_o2)
        h_n2 = calc_scale_height(mass_n2, alts, temp)
        # print('Scale height of n2 : ', h_n2)
    
        # calculate euvac (step 5):
        intensity_at_inf = EUVAC(euv_info['f74113'], euv_info['afac'], f107, f107a)
    
        # calculate mean wavelength:
        wavelength = (euv_info['short'] + euv_info['long'])/2
    
        # calculate energies (step 8):
        energies = convert_wavelength_to_joules(wavelength)
        
        # plot intensities at infinity:
        # plot_spectrum(wavelength, intensity_at_inf, 'intensity_inf.png')
    
        # Calculate the density of O as a function of alt and temp (step 6):
        density_o = calc_hydrostatic(n_o_bc, h_o, temp, alts)
        density_o2 = calc_hydrostatic(n_o2_bc, h_o2, temp, alts)
        density_n2 = calc_hydrostatic(n_n2_bc, h_n2, temp, alts)
        
        
        # plot out to a file:
        # plot_value_vs_alt(alts, density_o, 'o_init.png', '[O] (/m3)', is_log = True)
        # plot_value_vs_alt(alts, density_o2, 'o2_init.png', '[O2] (/m3)', is_log = True)
        # plot_value_vs_alt(alts, density_n2, 'n2_init.png', '[N2] (/m3)', is_log = True)
    
        # Calculate Taus for O (Step 7):
        if -75 < SZA < 75:
            tau_o = calc_tau_complete(SZA, density_o, h_o, euv_info['ocross'])
            tau_o2 = calc_tau_complete(SZA, density_o2, h_o2, euv_info['o2cross'])
            tau_n2 = calc_tau_complete(SZA, density_n2, h_n2, euv_info['n2cross'])
            tau = tau_o + tau_o2 + tau_n2 
            # print(np.shape(tau))
        else:
            tau_o = calc_tau_complete(SZA, density_o, h_o, euv_info['ocross'])
            tau = 10+tau_o*0
            # print(tau)
        
        # plot Tau to file:
        # for i in range(37):
        #     plot_value_vs_alt(alts, tau[i], 'tau_'+str(i)+'.png', 'tau ()')
        # plot_value_vs_alt(alts, tau[10], 'tau_'+str(5)+'.png', 'tau ()')
    
        Qeuv_o = calculate_Qeuv_complete(density_o,
                                intensity_at_inf,
                                tau,
                                euv_info['ocross'],
                                energies,
                                efficiency)
        
        Qeuv_o2 = calculate_Qeuv_complete(density_o2,
                                intensity_at_inf,
                                tau,
                                euv_info['o2cross'],
                                energies,
                                efficiency)
        
        Qeuv_n2 = calculate_Qeuv_complete(density_n2,
                                intensity_at_inf,
                                tau,
                                euv_info['n2cross'],
                                energies,
                                efficiency)
    
        Qeuv = Qeuv_o + Qeuv_o2 + Qeuv_n2 
    
        # plot out to a file:
        # plot_value_vs_alt(alts, Qeuv, 'qeuv.png', 'Qeuv (W/m3)')
        
        # alter this to calculate the real mass density (include N2 and O2):
        rho = calc_rho_complete(density_o, mass_o, density_o2, mass_o2, density_n2, mass_n2)
    
        # this provides cp, which could be made to be a function of density ratios:
        cp = calculate_cp()
        
        return Qeuv/(1500*rho)
    # dTdt = convert_Q_to_dTdt(Qeuv, rho, cp)

    # plot out to a file:
    # plot_value_vs_alt(alts, dTdt * 86400, 'dTdt.png', 'dT/dt (K/day)')
    
    
    "Solve for temperature"
    # initialize temperature (step 3):
    temp0 = init_temp(alts)
    
    # initialize time
    t0 = 0.0
    tf = 60*60*24
    time = np.arange(t0, tf, 5*60)
    
    # Initialize SZA
    SZA_vec = np.linspace(-180, 180, len(time))
    
    
    # temp_time = np.zeros((len(time),len(temp0)))
    temp = np.zeros((len(time), len(temp0)))
    temp[0] = temp0
    for j in range(len(time)-1):
        # NOTE: SVA NOT FULLY IMPLEMENTED. 
        # SZA = SZA_vec[j]
        temp_integrated = RK4_single_step(RHS, temp[j], time[j], 5*60)
        # print(temp_integrated)
        temp[j+1] = temp_integrated
    
    
    for j in range(len(time)):
        fig1 = plt.figure()
        plt.plot(temp[j,:],alts,'c-.',linewidth = 2)
        plt.grid()
        plt.ylabel('Altitude')
        plt.xlabel('Temperature')
        plt.legend(['rk4'])
