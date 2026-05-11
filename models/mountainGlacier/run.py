#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run script for the Glacier model

"""

# external library imports
from time import time
import os
import sys

import numpy as np ## added temp for max vels

# internal imports
sys.path.append("../../") # required so that we can find the central model code from here!
from solver.dataStructures import Grid
from solver.main import step
from setup import initializeModel, updateMarkers # updateGrid
from visualisation import makePlots

from solver.dataStructures import copyGrid
from solver.physics.grid_fns import basicGridVelocities
from visualisation import animateAVar
anim = 1



###############################################################################
# step 0 : start timer (for tracking performance) and set debug flags
###############################################################################
strt = time()

# if debugging, this should be 1 AND jitclass tags in solver/dataStructures and Parameters must be commented out!
os.environ["NUMBA_DISABLE_JIT"] = "0"
# if 1 prints out extra statements at various places in the timeloop
debug = 1


###############################################################################
# step 1 : initialize the model run
###############################################################################
params, grid, materials, markers, BC = initializeModel()


# initialize grid0 for old values
grid0 = Grid(grid.xnum, grid.ynum)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max


# if figures directory doesn't already exist, add it
if (os.path.exists(f"{params.output_path}/{params.output_name}")==False):
    os.makedirs(f"{params.output_path}/{params.output_name}")


if (anim):
    # for animation
    grid_list = []
    vxb_list = []
    vyb_list = []
    t_list = []

# Find velocity maxima/minima
vxmax_val = 0.
vxmin_val = 0.
vymax_val = 0.
vymin_val = 0.

###############################################################################
# step 2: time loop
###############################################################################
for nt in range(0, params.ntstp_max):
    
    ###########################################################################
    # do a timestep
    step(params, grid, materials, markers, BC, timestep, nt, grid0, debug)

    # check velocity maxima
    vxmax_0 = abs(np.max(grid.vx))
    vxmax_1 = abs(np.min(grid.vx))
    vymax_0 = abs(np.max(grid.vy))
    vymax_1 = abs(np.min(grid.vy))

    if vxmax_val < vxmax_0:
        vxmax_val = vxmax_0
    if vxmin_val < vxmax_1:
        vxmin_val = vxmax_1
    if vymax_val < vymax_0:
        vymax_val = vymax_0
    if vymin_val < vymax_1:
        vymin_val =  vymax_1

    vxmax = max(vxmax_0, vxmax_1)
    vymax = max(vymax_0, vymax_1)


    ###########################################################################
    # visualization
    if (nt%(params.save_fig)==0):
        print('plotting')

        # wrapper for calling whatever custom plots are defined in {model_name)/visualisations.py
        makePlots(grid, markers, params, nt, time_curr)

        if (anim):
            # interpolate vx, vy for basic grid
            vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)

            # Store data for animation
            grid_snapshot = Grid(grid.xnum, grid.ynum)
            copyGrid(grid, grid_snapshot)
            grid_list.append(grid_snapshot)
            vxb_list.append(vxb.copy())
            vyb_list.append(vyb.copy())
            t_list.append(time_curr / (365.25 * 24 * 3600))
        

    ###########################################################################
    # advance timestep
    
    time_curr += timestep
    print('Time: %.3f yr'%(time_curr/(365.25*24*3600)))
    # print('Time: %.3f Myr'%(time_curr*1e-6/(365.25*24*3600)))

    
    ###########################################################################
    # Make any model-specific adjustments 
    # eg. for a moving grid, update the grid positions
    # updateGrid(params, grid, time_curr, timestep, BC.B_bottom)
    updateMarkers(markers, params, grid)

    
        
    ###########################################################################
    # exit if final time is reached
    if (time_curr >= params.t_end):
        # we have reached the max time specified, exit loop
        print('t_end reached, exiting loop')
        break

end = time() - strt
print('time elapsed: %f'%(end))

print(f'Velocities: {vxmax_val*3600*24*365.25:1.3f}, {-vxmin_val*3600*24*365.25:1.3f}, {vymax_val*3600*24*365.25:1.3f}, {-vymin_val*3600*24*365.25:1.3f} [m/y]')

# visualization
print('plotting final timestep')

# wrapper for calling whatever custom plots are defined in {model_name)/visualisations.py
makePlots(grid, markers, params, nt, time_curr)

if (anim):
    # interpolate vx, vy for basic grid
    vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)

    # Store data for animation
    grid_snapshot = Grid(grid.xnum, grid.ynum)
    copyGrid(grid, grid_snapshot)
    grid_list.append(grid_snapshot)
    vxb_list.append(vxb.copy())
    vyb_list.append(vyb.copy())
    t_list.append(time_curr / (365.25 * 24 * 3600))

if (anim):
    # Save animation
    animateAVar(grid_list, vxb_list, vyb_list, params, t_list, filename='glacier_animation.gif')