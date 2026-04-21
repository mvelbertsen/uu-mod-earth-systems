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
sys.path.append("../../") # required so that we can find the central model code from here!
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
params, grid, materials, markers, BC = initializeModel()


# initialize grid0 for old values
grid0 = Grid(grid.xnum, grid.ynum)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max


# if figures directory doesn't already exist, add it
if (os.path.exists(f"{params.output_path}/{params.output_name}")==False):
    os.makedirs(f"{params.output_path}/{params.output_name}")
    

###############################################################################
# step 2: time loop
###############################################################################
for nt in range(0, params.ntstp_max):
    
    ###########################################################################
    # do a timestep
    step(params, grid, materials, markers, BC, timestep, nt, grid0, debug)


    ###########################################################################
    # visualization
    if (nt%(params.save_fig)==0):
        print('plotting')

        # wrapper for calling whatever custom plots are defined in {model_name)/visualisations.py
        makePlots(grid, markers, params, nt, time_curr)
        

    ###########################################################################
    # advance timestep
    
    time_curr += timestep
    print('Time: %.3f Myr'%(time_curr*1e-6/(365.25*24*3600)))

    
    ###########################################################################
    # Make any model-specific adjustments 
    # eg. for a moving grid, update the grid positions
    updateGrid(params, grid, time_curr, timestep, BC.B_bottom)
    
        
    ###########################################################################
    # exit if final time is reached
    if (time_curr >= params.t_end):
        # we have reached the max time specified, exit loop
        print('t_end reached, exiting loop')
        break

end = time() - strt
print('time elapsed: %f'%(end))