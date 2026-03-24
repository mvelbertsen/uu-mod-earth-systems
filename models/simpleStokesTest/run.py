#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example script for running model w. simple Stokes solver test

"""

# external library imports
from time import time
import os
import sys

# internal imports
sys.path.append("../../") # required so that we can find the rest of the code from here

from solver.dataStructures import Grid
from solver.main import step
from setup import initializeModel, gridSpacings
from solver.physics.grid_fns import basicGridVelocities
from visualisation import makePlots



###############################################################################
# step 0 : start timer (for tracking performance) and set debug flags
###############################################################################
strt = time()

# if debugging, this should be 1 AND jitclass tags in solver/dataStructures must be commented out!
os.environ["NUMBA_DISABLE_JIT"] = "0"
# if 1 prints out extra statements at various places in the timeloop
debug = 1


###############################################################################
# step 1 : initialize the model run
###############################################################################
params, grid, materials, markers, P_first, B_top, B_bottom,\
B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right = initializeModel()

# initialize grid0 for old values
grid0 = Grid(grid.xnum, grid.ynum)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max

# if figures directory doesn't already exist, add it
if (os.path.exists("figures/")==False):
    os.mkdir("figures/")


###############################################################################
# step 2: time loop
###############################################################################
for nt in range(0, params.ntstp_max):
    
    ###########################################################################
    # do a timestep
    step(params, grid, materials, markers, P_first, B_top, B_bottom,\
    B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right, timestep, nt, grid0, debug)


    ###########################################################################
    # visualization + output    
    if (nt%(params.save_fig)==0):
        print('plotting')
        
        # interpolate vx, vy for basic grid
        vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)
        
        makePlots(grid, vxb, vyb, params, nt, time_curr)
        

    ###########################################################################
    # advance timestep
    
    time_curr += timestep
    print('Time: %.3f Myr'%(time_curr*1e-6/(365.25*24*3600)))
    
    
    ###########################################################################
    # Make any model-specific adjustments 
    # eg. for a moving grid, update the grid positions
    
    if (debug):
        print('updating grid spacings')
    
    # update grid positions based on extension
    params.ysize += -params.v_ext/params.xsize*params.ysize*timestep
    params.xsize += params.v_ext*timestep
    
    g = gridSpacings(params, grid, time_curr)
    
    # if we have changing grid, need to update bottom BC
    if (abs(params.v_ext)>0):
        B_bottom[:,2] = -params.v_ext/params.xsize*params.ysize
        B_bottom[:,3] = 0
    
        
    ###########################################################################
    # exit if final time is reached
    if (time_curr >= params.t_end):
        # we have reached the max time specified, exit loop
        print('t_end reached, exiting loop')
        break


end = time() - strt
print('time elapsed: %f'%(end)) 