#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run script for the Subduction model

"""

# external library imports
from time import time
import os
import sys

# internal imports
sys.path.append("../../") # required so that we can find the rest of the code from here!
from solver.dataStructures import Grid
from solver.main import step
from setup import initializeModel, updateGrid
from visualisation import makePlots


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
params, grid, materials, markers, P_first, B_top, B_bottom,\
B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right = initializeModel()


# initialize grid0 for old values
grid0 = Grid(grid.xnum, grid.ynum)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max

# create path to write directory

# if figures directory doesn't already exist, add it
if (os.path.exists(f"{params.output_path}/{params.output_name}")==False):
    os.makedirs(f"{params.output_path}/{params.output_name}")
    
    

#TODO output initial state with makePlots


###############################################################################
# step 2: time loop
###############################################################################
for nt in range(0, params.ntstp_max):
    
    ###########################################################################
    # do a timestep
    step(params, grid, materials, markers, P_first, B_top, B_bottom,\
    B_left, B_right, B_intern, BT_top, BT_bottom, BT_left, BT_right, timestep, nt, grid0, debug)


    ###########################################################################
    # visualization
    if (nt%(params.save_fig)==0):
        print('plotting')
        
        # interpolate vx, vy for basic grid
        # vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)
        
        # wrapper for calling whatever custom plots are defined in setup.py
        makePlots(grid, markers, params, nt, time_curr)
        

    ###########################################################################
    # advance timestep
    
    time_curr += timestep
    print('Time: %.3f Myr'%(time_curr*1e-6/(365.25*24*3600)))

    
    ###########################################################################
    # Make any model-specific adjustments 
    # eg. for a moving grid, update the grid positions
    updateGrid(params, grid, time_curr, timestep, B_bottom)
    
        
    ###########################################################################
    # exit if final time is reached
    if (time_curr >= params.t_end):
        # we have reached the max time specified, exit loop
        print('t_end reached, exiting loop')
        break

end = time() - strt
print('time elapsed: %f'%(end)) 