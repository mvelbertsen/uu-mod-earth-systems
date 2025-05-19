#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test of the Temperature solver fn, using simple setup with square of higher T 
in the center of the domain
 
"""
import numpy as np
import matplotlib.pyplot as plt
from Temperature_solver_grid import Temperature_solver_grid

# set grid parameters
xres = 51
yres = 31

L_x = 1e6 # m, physical size x
L_y = 1.5e6 # m, physical size y

dx = L_x/(xres-1)
dy = L_y/(yres-1)

# spatial positions for basic grid points
x = np.linspace(0,L_x,xres)
y = np.linspace(0,L_y,yres)

# spatial positions of P grid points
x_p = np.linspace(-dx/2,L_x+dx/2,xres+1)
y_p = np.linspace(-dy/2,L_y+dy/2,yres+1)

# create thermal properties arrays
k_c = np.ones([yres,xres])*3.0 # thermal conductivity, W m^-1
rho = np.ones([yres,xres])*3200.0 # density, kg m^-3
rho_C_P = np.ones([yres,xres])*1000.0*3200 # heat capacity, J kg^-1 K^-1

# create array for temperature 
T_b = 1000.0
T_w = 1300.0
T = np.ones([yres,xres])*T_b

T_BC = T_b

# fill with initial conditions
for j in range(0,xres):
    for i in range(0,yres):
        if (y[i]>L_y*0.3 and y[i] < L_y*0.7 and x[j] > L_x*0.3 and x[j] < L_x*0.7):
            T[i,j] = T_w
            k_c[i,j] = 10.0
            rho_C_P[i,j] = 3300.0*1100

# also for storing previous T
T_old = np.copy(T)

# create arrays for linear system
N = xres*yres # total number of unknowns
L = np.zeros([N,N])
R = np.zeros([N])

# time stepping params
n_tsteps = 20 # number of timesteps
t_el = 0 # elapsed time, s

# calc for the max thermal diffusivity
kappa_mx = 10/(3200*1000)
dt = min(dx,dy)**2/(3*kappa_mx)

# rheet
R_heat = np.zeros([yres, xres])
# BCs
# consant T
B_top = np.zeros([xres, 2])
B_bottom = np.zeros([xres, 2])
B_left = np.zeros([yres, 2])
B_right = np.zeros([yres, 2])

B_top[:,0] = T_BC
B_bottom[:,0] = T_BC
B_left[:,0] = T_BC
B_right[:,0] = T_BC

# loop over number of timesteps
for t in range(0,n_tsteps):
    
    T_new, T_res = Temperature_solver_grid(dt, xres, yres, x, y, k_c, rho_C_P, B_top, B_bottom, B_left, B_right, R_heat, T_old)
    
    # store the new value as T_old
    T_old = np.copy(T_new)
            
    t_el += dt
            
    # plot the result
    fig, axs = plt.subplots(1,1, figsize=(9,9), constrained_layout=True)
    im = axs.imshow(T_old, origin='upper', aspect='auto', extent=[0,L_x,L_y,0],vmin=T_b ,vmax=T_w )
    fig.colorbar(im, ax=axs,pad=0.0) # display colorbar
    axs.set_title('Temperature after %.1e yr'%(t_el/365.25/24/60/60))     # set plot title
    axs.set(xlabel='x (km)',ylabel='y (km)')         # label the y-axis (shared axis for x)
    plt.show()
    
    
    
    