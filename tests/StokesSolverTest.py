#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test case for the StokesContinuity solver function
Viscosity, density is different in the two halves of the domain.
"""

import numpy as np
import matplotlib.pyplot as plt

from StokesContinuitySolver import Stokes_Continuity_solver


# set grid parameters
xres = 31
yres = 21

L_x = 1e5 # m, physical size x
L_y = 1.5e5 # m, physical size y

dx = L_x/(xres-1)
dy = L_y/(yres-1)

# spatial positions for basic grid points
x = np.linspace(0,L_x,xres)
y = np.linspace(0,L_y,yres)

# set up initial conditions
# pressure

# viscosity
eta = np.zeros([yres,xres])
# fill with the visc structure
eta[:,0:int(xres/2)] = 1e20
eta[:,int(xres/2):] = 1e20

# try setting shear and normal stresses to the same

eta_n = np.zeros([yres-1,xres-1])
for j in range(0,xres-1):
    for i in range(0,yres-1):
        eta_n[i,j] = 1/((1/eta[i,j] + 1/eta[i,j+1] + 1/eta[i+1,j] + 1/eta[i+1,j+1])/4)

# grav acc
gy = 10

# density
rho = np.zeros([yres,xres])

# fill with the dens structure
rho[:,0:int(xres/2)] = 3200
rho[:,int(xres/2):] = 3300

#rho[:,:] = 3200
#rho[5:15,10:20] = 3300

RX1 = np.zeros([yres+1,xres])
# y-Stokes
RY1 = np.zeros([yres,xres+1])
# continuity
RC1 = np.zeros([yres-1,xres-1])
# Grid points cycle
for i in range(1,yres):
    for j in range(1, xres):
        # Right part of x-Stokes Equation is zero! as is continuity
        if(i<yres-1):
            RY1[i,j] = -gy*(rho[i,j] + rho[i,j-1])/2
            
# pressure scaling/ BCS
Pfirst = np.array([0,0])

# boundary conditions ??? Apparently this should give free slip?
B_top = np.zeros([xres+1,4])
B_top[:,1] = 1
B_bottom = np.zeros([xres+1,4])
B_bottom[:,1] = 1
B_left = np.zeros([yres+1,4])
B_left[:,3] = 1
B_right = np.zeros([yres+1,4])
B_right[:,3] = 1

# optional internal boundary, switched off
B_intern = np.zeros([8])
B_intern[0] = -1
B_intern[4] = -1


vx, vy, P, resx, resy, resc,L,R = Stokes_Continuity_solver(Pfirst, eta, eta_n, xres, yres, x, y, RX1, RY1, RC1, B_top, B_bottom, B_left, B_right, B_intern)

vx_min = -5e-10
vx_max = 5e-10

vy_min = -5e-10
vy_max = 5e-10


fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(18,18), constrained_layout=True)

# plot the density as colormap
im = axs[0,0].imshow(rho, origin='upper', aspect='auto', extent=[0,L_x,L_y,0])
fig.colorbar(im, ax=axs[0,0],pad=0.0) # display colorbar
axs[0,0].set_title('Viscosity')     # set plot title
axs[0,0].set(ylabel='y (km)')         # label the y-axis (shared axis for x)
qu = axs[0,0].quiver(x,np.flip(y,0),np.flip(-vx[:yres,:],1),np.flip(-vy[:,:xres],0)) 
#qu = axs[0,0].quiver(x,y,-vx[:yres,:],-vy[:,:xres]) 

# plot the pressure as colormap
im = axs[0,1].imshow(P[1:yres,1:xres], origin='upper', aspect='auto', extent=[0,L_x,L_y,0])
fig.colorbar(im, ax=axs[0,1],pad=0.0) # display colorbar
axs[0,1].set_title('Pressure')     # set plot title

# plot vx as colormap
im = axs[1,0].imshow(vx, origin='upper', aspect='auto', extent=[0,L_x,L_y,0])#,vmin=vx_min,vmax=vx_max)
fig.colorbar(im, ax=axs[1,0],pad=0.0) # display colorbar
axs[1,0].set_title('Vx')     # set plot title
axs[1,0].set(ylabel='y (km)', xlabel='x (km)')         # label the y-axis (shared axis for x)

# plot vy as colormap
im = axs[1,1].imshow(vy, origin='upper', aspect='auto', extent=[0,L_x,L_y,0])#,vmin=vy_min,vmax=vy_max)
fig.colorbar(im, ax=axs[1,1],pad=0.0) # display colorbar
axs[1,1].set_title('Vy')     # set plot title
axs[1,1].set(xlabel='x (km)')         

plt.savefig('Stokes_varVisc.png')
plt.show()














